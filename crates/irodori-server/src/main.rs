//! irodori-server: irodori-tts as a child process.
//!
//! The host application starts this executable, sends requests on stdin and reads responses from stdout, and ends it by
//! killing it or closing stdin. Running out of process has two reasons: an application that already links another ONNX
//! Runtime binding cannot link the `ort` crate in the same Cargo graph (both declare `links = "onnxruntime"`), and killing
//! the process is the one way to give back all the GPU memory (CUDA context included).
//!
//!   irodori-server --dir <root>
//!
//! `<root>` holds `models/` (the fp16 graphs), `tokenizer/` (tokenizer.json, tokenizer_config.json) and `runtime/`
//! (`irodori_onnxruntime.dll` and the CUDA provider with its NVIDIA DLLs).
//!
//! A frame is `u32 LE header length, header (JSON), u32 LE body length, body`, in both directions. Requests are answered
//! one by one, in order. Responses carry `{"ok": true, ...}` or `{"ok": false, "error": "..."}` in the header.
//!
//!   {"cmd": "load"}                                          load the synthesis engine (idempotent)
//!   {"cmd": "synthesize", "text", "voice": <.irvc path>, "duration_scale"?, "seed"?, "max_seconds"?, "fade_out_ms"?, "trim_leading_silence"?, "trim_trailing_silence"?, "caption"?}
//!                                     -> header also carries "predicted_seconds" (see Synthesis::predicted_seconds), body: a 16-bit mono WAV
//!   {"cmd": "register", "out": <.irvc path>, "clips": [{"rate", "samples"}...], "max_seconds"}
//!                                                            body: 16-bit little-endian mono PCM of all clips back to back
//!   {"cmd": "register_no_reference", "out": <.irvc path>}    no body — a Voice with no speaker
//!                                                            conditioning at all (see VoiceRegistrar::register_no_reference)
//!
//! stdout carries nothing but frames; diagnostics go to stderr.

use std::io::{self, Cursor, Read, Write};
use std::path::PathBuf;

use irodori_tts::{Clip, Engine, EngineConfig, SynthOptions, Voice, VoiceRegistrar};
use serde_json::{json, Value};

fn read_frame(input: &mut impl Read) -> io::Result<Option<(Value, Vec<u8>)>> {
    let mut len = [0u8; 4];
    match input.read_exact(&mut len) {
        Ok(()) => {}
        // The host closed stdin: normal shutdown.
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let mut header = vec![0u8; u32::from_le_bytes(len) as usize];
    input.read_exact(&mut header)?;
    input.read_exact(&mut len)?;
    let mut body = vec![0u8; u32::from_le_bytes(len) as usize];
    input.read_exact(&mut body)?;
    Ok(Some((serde_json::from_slice(&header)?, body)))
}

fn write_frame(output: &mut impl Write, header: &Value, body: &[u8]) -> io::Result<()> {
    let header = serde_json::to_vec(header)?;
    output.write_all(&(header.len() as u32).to_le_bytes())?;
    output.write_all(&header)?;
    output.write_all(&(body.len() as u32).to_le_bytes())?;
    output.write_all(body)?;
    output.flush()
}

fn wav_bytes(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, String> {
    let spec = hound::WavSpec { channels: 1, sample_rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut cursor, spec).map_err(|e| e.to_string())?;
    for &s in samples {
        writer.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16).map_err(|e| e.to_string())?;
    }
    writer.finalize().map_err(|e| e.to_string())?;
    Ok(cursor.into_inner())
}

struct Server {
    config: EngineConfig,
    engine: Option<Engine>,
}

impl Server {
    // The Ok side is (extra response-header fields, body) — only "synthesize" has anything to
    // report beyond "ok": true (predicted_seconds; see Synthesis::predicted_seconds' own comment).
    fn handle(&mut self, request: &Value, body: &[u8]) -> Result<(Value, Vec<u8>), String> {
        let text_field = |name: &str| request[name].as_str().ok_or_else(|| format!("missing \"{name}\""));
        match request["cmd"].as_str() {
            Some("load") => {
                if self.engine.is_none() {
                    self.engine = Some(Engine::new(&self.config).map_err(|e| e.to_string())?);
                }
                Ok((json!({}), Vec::new()))
            }
            Some("synthesize") => {
                let voice = Voice::load(text_field("voice")?).map_err(|e| e.to_string())?;
                let mut options = SynthOptions::default();
                if let Some(scale) = request["duration_scale"].as_f64() {
                    options.duration_scale = scale as f32;
                }
                if let Some(seed) = request["seed"].as_u64() {
                    options.seed = seed;
                }
                if let Some(max_seconds) = request["max_seconds"].as_f64() {
                    options.max_seconds = max_seconds as f32;
                }
                if let Some(ms) = request["fade_out_ms"].as_f64() {
                    options.fade_out_ms = ms as f32;
                }
                if let Some(trim) = request["trim_leading_silence"].as_bool() {
                    options.trim_leading_silence = trim;
                }
                if let Some(trim) = request["trim_trailing_silence"].as_bool() {
                    options.trim_trailing_silence = trim;
                }
                if let Some(caption) = request["caption"].as_str() {
                    options.caption = Some(caption.to_string());
                }
                let engine = self.engine.as_mut().ok_or("the engine is not loaded")?;
                let out = engine.synthesize(text_field("text")?, &voice, &options).map_err(|e| e.to_string())?;
                let wav = wav_bytes(&out.samples, out.sample_rate)?;
                Ok((json!({ "predicted_seconds": out.predicted_seconds }), wav))
            }
            Some("register") => {
                let mut budget = request["max_seconds"].as_f64().ok_or("missing \"max_seconds\"")? as f32;
                let mut offset = 0;
                let mut recordings: Vec<(Vec<f32>, u32)> = Vec::new();
                for clip in request["clips"].as_array().ok_or("missing \"clips\"")? {
                    let rate = clip["rate"].as_u64().ok_or("clip without \"rate\"")? as u32;
                    let count = clip["samples"].as_u64().ok_or("clip without \"samples\"")? as usize;
                    let pcm = body.get(offset..offset + count * 2).ok_or("the body is shorter than the clips say")?;
                    offset += count * 2;
                    // Clips past the budget are skipped, the one crossing it is cut.
                    let keep = count.min((budget.max(0.0) * rate as f32) as usize);
                    if keep == 0 {
                        continue;
                    }
                    budget -= keep as f32 / rate as f32;
                    let samples = pcm[..keep * 2].chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0).collect();
                    recordings.push((samples, rate));
                }
                // The registrar keeps two graphs on the GPU, so it only lives as long as this registration.
                let mut registrar = VoiceRegistrar::new(&self.config).map_err(|e| e.to_string())?;
                let clips: Vec<Clip> = recordings.iter().map(|(samples, rate)| Clip { samples, sample_rate: *rate }).collect();
                let voice = registrar.register(&clips).map_err(|e| e.to_string())?;
                voice.save(text_field("out")?).map_err(|e| e.to_string())?;
                Ok((json!({}), Vec::new()))
            }
            Some("register_no_reference") => {
                let mut registrar = VoiceRegistrar::new(&self.config).map_err(|e| e.to_string())?;
                let voice = registrar.register_no_reference().map_err(|e| e.to_string())?;
                voice.save(text_field("out")?).map_err(|e| e.to_string())?;
                Ok((json!({}), Vec::new()))
            }
            other => Err(format!("unknown command {other:?}")),
        }
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir: PathBuf = args.iter().position(|a| a == "--dir").and_then(|i| args.get(i + 1)).expect("usage: irodori-server --dir <root>").into();
    // cuDNN loads its own component DLLs (cudnn_graph64_9.dll, ...) with the default search order, which does not include the
    // directory of the DLL doing the loading; without this it fails with "Could not locate cudnn_graph64_9.dll".
    let runtime = dir.join("runtime");
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::set_var("PATH", std::env::join_paths(std::iter::once(runtime).chain(std::env::split_paths(&path))).unwrap());
    let mut server = Server {
        config: EngineConfig {
            model_dir: dir.join("models"),
            tokenizer_dir: dir.join("tokenizer"),
            ort_dylib: Some(dir.join("runtime").join("irodori_onnxruntime.dll")),
            fp16: true,
            cuda_device: Some(0),
        },
        engine: None,
    };
    let (mut input, mut output) = (io::stdin().lock(), io::stdout().lock());
    while let Some((request, body)) = read_frame(&mut input)? {
        let (header, body) = match server.handle(&request, &body) {
            Ok((mut extra, body)) => {
                extra["ok"] = json!(true);
                (extra, body)
            }
            Err(error) => (json!({ "ok": false, "error": error }), Vec::new()),
        };
        write_frame(&mut output, &header, &body)?;
    }
    Ok(())
}

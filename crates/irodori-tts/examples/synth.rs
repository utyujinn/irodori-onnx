//! Small command line front end.
//!
//!   synth register <voice.irvc> <recording.wav>...   register a reference voice from one or more WAV files
//!   synth say <voice.irvc> "<text>" <out.wav> [--seed N] [--steps N] [--repeat N] [--caption "<style>"]
//!   synth bench <voice.irvc> <lines.txt>              synthesize every line twice and report the timing of both passes
//!                                                     (pass 1 sees every input length for the first time)
//!
//! Configuration through environment variables:
//!   IRODORI_MODEL_DIR (ONNX graphs), IRODORI_TOKENIZER_DIR (checkpoint tokenizer/ folder),
//!   optional IRODORI_ORT_DLL, IRODORI_FP16=1, IRODORI_CUDA=<device id>.
//!
//! Only register voices you have the right to use; see NOTICE.md about the model's ethical restrictions.

use std::time::Instant;

use irodori_tts::{Clip, Engine, EngineConfig, SynthOptions, Voice, VoiceRegistrar};

fn config() -> EngineConfig {
    let var = |name: &str| std::env::var(name).unwrap_or_else(|_| panic!("set {name}"));
    EngineConfig {
        model_dir: var("IRODORI_MODEL_DIR").into(),
        tokenizer_dir: var("IRODORI_TOKENIZER_DIR").into(),
        ort_dylib: std::env::var("IRODORI_ORT_DLL").ok().map(Into::into),
        fp16: std::env::var("IRODORI_FP16").is_ok_and(|v| v == "1"),
        cuda_device: std::env::var("IRODORI_CUDA").ok().and_then(|v| v.parse().ok()),
    }
}

/// Reads a WAV file and mixes it down to mono f32.
fn read_wav(path: &str) -> (Vec<f32>, u32) {
    let mut reader = hound::WavReader::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let spec = reader.spec();
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(Result::unwrap).collect(),
        hound::SampleFormat::Int => {
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.unwrap() as f32 / scale).collect()
        }
    };
    let channels = spec.channels as usize;
    let mono = interleaved.chunks_exact(channels).map(|frame| frame.iter().sum::<f32>() / channels as f32).collect();
    (mono, spec.sample_rate)
}

fn write_wav(path: &str, samples: &[f32], sample_rate: u32) {
    let spec = hound::WavSpec { channels: 1, sample_rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for &s in samples {
        writer.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16).unwrap();
    }
    writer.finalize().unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("register") if args.len() >= 3 => {
            let clips: Vec<(Vec<f32>, u32)> = args[2..].iter().map(|p| read_wav(p)).collect();
            let started = Instant::now();
            let mut registrar = VoiceRegistrar::new(&config()).unwrap();
            let loaded = started.elapsed();
            let voice = registrar
                .register(&clips.iter().map(|(samples, sample_rate)| Clip { samples, sample_rate: *sample_rate }).collect::<Vec<_>>())
                .unwrap();
            voice.save(&args[1]).unwrap();
            println!("registered {} tokens from {} recording(s) in {:.2}s (models loaded in {:.2}s) -> {}", voice.tokens(), clips.len(), (started.elapsed() - loaded).as_secs_f32(), loaded.as_secs_f32(), args[1]);
        }
        Some("say") if args.len() >= 4 => {
            let option = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|v| v.parse::<usize>().ok());
            let string_option = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
            let options = SynthOptions {
                seed: option("--seed").unwrap_or(0) as u64,
                steps: option("--steps").unwrap_or(4),
                caption: string_option("--caption"),
                ..SynthOptions::default()
            };
            let voice = Voice::load(&args[1]).unwrap();
            let started = Instant::now();
            let mut engine = Engine::new(&config()).unwrap();
            println!("models loaded in {:.2}s", started.elapsed().as_secs_f32());
            // The first runs include one-time setup, so the timing of every run is printed.
            let mut last = None;
            for run in 0..option("--repeat").unwrap_or(1) {
                let started = Instant::now();
                let out = engine.synthesize(&args[2], &voice, &SynthOptions { seed: options.seed + run as u64, ..options.clone() }).unwrap();
                let took = started.elapsed().as_secs_f32();
                let seconds = out.samples.len() as f32 / out.sample_rate as f32;
                println!("run {run}: {took:.2}s for {seconds:.2}s of audio (RTF {:.2})", took / seconds);
                last = Some(out);
            }
            let out = last.unwrap();
            write_wav(&args[3], &out.samples, out.sample_rate);
            println!("wrote {}", args[3]);
        }
        Some("bench") if args.len() >= 3 => {
            let voice = Voice::load(&args[1]).unwrap();
            let lines: Vec<String> = std::fs::read_to_string(&args[2]).unwrap().lines().filter(|l| !l.trim().is_empty()).map(str::to_string).collect();
            let mut engine = Engine::new(&config()).unwrap();
            engine.synthesize("ウォームアップです。", &voice, &SynthOptions::default()).unwrap();
            for pass in 1..=2 {
                let mut times: Vec<f32> = lines
                    .iter()
                    .map(|line| {
                        let started = Instant::now();
                        engine.synthesize(line, &voice, &SynthOptions::default()).unwrap();
                        started.elapsed().as_secs_f32()
                    })
                    .collect();
                let mean = times.iter().sum::<f32>() / times.len() as f32;
                times.sort_by(f32::total_cmp);
                println!("pass {pass}: median {:.2}s mean {mean:.2}s min {:.2}s max {:.2}s ({} lines)", times[times.len() / 2], times[0], times[times.len() - 1], times.len());
            }
        }
        _ => eprintln!("usage:\n  synth register <voice.irvc> <recording.wav>...\n  synth say <voice.irvc> \"<text>\" <out.wav> [--seed N] [--steps N] [--repeat N]
  synth bench <voice.irvc> <lines.txt>"),
    }
}

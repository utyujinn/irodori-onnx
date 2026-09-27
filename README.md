# irodori-onnx

Run [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) (v4.1 MeanFlow, Japanese TTS with zero-shot voice cloning and
emoji-based style control) **without Python or PyTorch**: convert it to ONNX, then run it from Rust on ONNX Runtime (CUDA).

> **Status: works, not yet polished.** The export pipeline was re-run end to end from a clean checkout by following this README,
> and the Rust crate (`crates/irodori-tts`) registers reference voices and synthesizes speech. Its output is checked against the
> Python runtime by tests (see [Tests](#tests)). The crate is not published on crates.io and its API may still change.
> Comments inside the scripts are in Japanese.

Please read [NOTICE.md](NOTICE.md) first: the model has **ethical restrictions** (no voice cloning without consent,
no deceptive deepfakes), and this project does **not** implement the SilentCipher watermark of the original runtime.

## Results

Model: `Aratako/Irodori-TTS-v4.1-Small-MF`, 4 sampling steps. Measured on Windows 11, RTX 4070 (12 GB), with VRChat running.

| Item | Result |
|---|---|
| Export | All 6 parts export with fully dynamic lengths: text encoder, speaker encoder, duration predictor, DiT step, codec decoder, codec encoder |
| Parity vs PyTorch (ONNX Runtime CPU) | relative L2 error 1e-6 to 1e-7 per part; end-to-end waveform correlation 1.000000 |
| Parity, fp16 on CUDA | end-to-end waveform correlation 0.9996 to 0.982 depending on the voice (one listener heard almost no difference) |
| **Rust runtime, fp16 on CUDA** | **24 different sentences, each length seen for the first time: median 0.50 s, max 0.65 s** (second pass: 0.49 s, so there is no first-time penalty). The Python server (same GPU, same setup) takes about 0.8 s. Registering a voice from 3 recordings takes 0.86 s |
| VRAM, Rust process | about 1 GB after loading, **2.5 GB peak** while synthesizing (the Python server: 3.4 GB resident, 3.7 GB peak) |
| Size (fp16) | 1.74 GB of ONNX (DiT 749 MB, text 636 MB, speaker 123 MB, decoder 131 MB, encoder 56 MB, duration 44 MB) |
| Runtime files | `onnxruntime_providers_cuda.dll` 349 MB + minimal NVIDIA runtime 1.84 GB (cuFFT is required; cuDNN `adv`/`runtime_compiled` and NVRTC are not). About 3.1 GB when compressed |

## Things that were not obvious (please keep them in mind if you port other models)

1. **Complex RoPE cannot be exported.** Replace `precompute_freqs_cis` / `apply_rotary_emb` with a real-valued
   (cos, sin) implementation; the outputs are bit-identical (`common.patch_real_rope`).
2. **`int(query_len)` in `build_context_attention_plan` specializes the sequence length** to the example input.
   Keep the symbolic size instead (see `02_export_dit_step.py`).
3. **Data-dependent branches** such as `if bool(has_any.all())` in `_safe_attention_mask` break `torch.export`; use a
   branch-free equivalent (`03_export_rest.py`).
4. **fp16 conversion breaks the DiT and the speaker encoder.** RMSNorm's `x * x` reaches 7e4 to 2.5e6, above fp16's 65504,
   and turns into `inf`. `06_fp16.py` keeps the RMSNorm mean-square path (Mul, ReduceMean, Add, Sqrt, Reciprocal) in fp32
   and then **removes the fp32 -> fp16 -> fp32 round-trip casts** that the converter inserts around every blocked node
   (otherwise the values are rounded to fp16 between two fp32 nodes and overflow anyway).
5. **Compute the sinusoidal timestep embedding outside the graph.** Its phase reaches about 1000 rad, which fp16 cannot
   represent; `02_export_dit_step.py` takes the embedding as an input.
6. **The empty caption is a constant.** With an empty caption the runtime passes an all-masked BOS token, so the caption
   state is all zeros and the caption encoder never needs to run.
7. **Text does not need padding to 256 tokens.** The Python runtime pads every request to 256 tokens; the ONNX graphs take
   the real length.
8. **Set `cudnn_conv_algo_search=HEURISTIC` on the ONNX Runtime CUDA EP.** With the default, a first run of the codec decoder
   did not finish in over 15 minutes; HEURISTIC fixed it (presumably the default's exhaustive per-shape convolution search).
9. **Keep the VRAM down with two settings.** A `RunOptions` with `memory.enable_memory_arena_shrinkage = "gpu:0"` (about
   1.1 GB less peak, no measurable slowdown) and `arena_extend_strategy = SameAsRequested` on the CUDA EP (another 1 GB in the
   Rust process). Without them ONNX Runtime behaves like the Python server or worse.
10. **Crate / library pairing:** the `ort` crate `2.0.0-rc.11` targets ONNX Runtime 1.23. `rc.13` targets 1.28 and does not
    load a 1.23 DLL. If your app already ships another ONNX Runtime, load this one under a different file name
    (verified: it coexists with sherpa-onnx's and VOICEVOX's copies in the same process).
11. `onnx.save(..., all_tensors_to_one_file=True)` **appends** to an existing external-data file; delete it first.
12. **The codec encoder must not pad inside the graph.** The original `_pad` is `if length % hop: pad`, and exporting it
    freezes the branch, so a recording whose length is an exact multiple of the hop (1920 samples) gets one extra latent
    frame. The exported encoder expects a length that is already a multiple of the hop; the Rust code pads.
13. **Resample exactly like the Python pipeline.** The crate implements `torchaudio.functional.resample` (Hann-windowed sinc,
    width 6, rolloff 0.99): relative error 3e-6 against torchaudio. A general FFT resampler crate (rubato 5) produced a
    glitch of 640 samples at the start of the clip and a 12 % error in the resulting latent.
14. **Loudness normalization differs slightly.** The `ebur128` crate and audiotools measured -16.575 and -16.617 LUFS for the
    same signal (0.04 dB), which shows up as about 1e-3 in the latent and 5e-3 in the speaker state.

## Layout

```text
export/            Python: ONNX export, fp16 conversion, parity data and tools
  common.py                    model loading, real-RoPE patch, real conditions for parity data
  make_latents.py              reference wav -> latent (.pt) using the exact preprocessing of the Python runtime
  01_make_step_reference.py    ground-truth data for one DiT step
  02_export_dit_step.py        DiT step (graph name: dit_step2)
  03_export_rest.py            text encoder, speaker encoder, duration predictor, codec decoder
  04_export_codec_encoder.py   codec encoder (used when registering a reference voice)
  05_e2e_onnx.py               full synthesis from ONNX only, compared with PyTorch; also stage timings
  06_fp16.py                   RMSNorm-safe fp16 conversion
  09_bisect_fp16.py            finds the first operator whose fp16 output diverges from fp32
  10_min_cuda.py               checks which NVIDIA DLLs are really required
  11_vram.py                   per-process VRAM and timing (Windows GPU performance counters)
  12_golden_text.py            ground truth for text normalization, tokens and duration features
  13_golden_synth.py           ground truth for a whole synthesis and for voice registration (synthetic reference signal)
crates/irodori-tts/  Rust runtime
  src/text.rs                  normalization and duration features (ported from Python)
  src/tokenizer.rs             ModernBERT-ja tokenizer.json + BOS
  src/engine.rs                the synthesis pipeline on ONNX Runtime
  src/registrar.rs, audio.rs   reference voice registration: resample, loudness, codec encoder, speaker encoder
  src/sampler.rs, postprocess.rs, voice.rs
  examples/synth.rs            command line: register a voice, say a sentence, benchmark
  tests/                       parity tests against the Python ground truth (tests/data/*.json)
```

## Reproducing the export (Windows, Python 3.10, CUDA 12.8)

```text
uv venv --python 3.10 .venv
uv pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install --override export/overrides.txt --prerelease=allow -r export/requirements.txt

# put your own reference recording (mono, 44.1 kHz) at export/voices/ref1.wav, then:
python export/make_latents.py
python export/01_make_step_reference.py
python export/02_export_dit_step.py
python export/03_export_rest.py
python export/04_export_codec_encoder.py
python export/06_fp16.py dit_step2 speaker_encoder text_encoder duration codec_decoder codec_encoder
ONNX_SUFFIX=_fp16 python export/05_e2e_onnx.py cuda     # parity and timings
```

Environment variables: `IRODORI_VOICES_DIR` (default `export/voices`), `IRODORI_REF_VOICES` (comma-separated reference names,
default `ref1`). Only record or use voices you have the right to use. The graphs end up in `export/ref/`.

## Using the Rust crate

Windows, an NVIDIA GPU, an ONNX Runtime 1.23 DLL with the CUDA provider, and the NVIDIA runtime DLLs on `PATH`.

```text
set IRODORI_MODEL_DIR=<export/ref>
set IRODORI_TOKENIZER_DIR=<checkpoint folder>\tokenizer
set IRODORI_ORT_DLL=<path to onnxruntime.dll, ideally renamed>
set IRODORI_FP16=1
set IRODORI_CUDA=0

cargo run --release --example synth -- register my.irvc recording1.wav recording2.wav
cargo run --release --example synth -- say my.irvc "おつかれさまです。" out.wav
cargo run --release --example synth -- bench my.irvc lines.txt
```

In code: create a `VoiceRegistrar` once per voice (drop it afterwards to give the VRAM back), keep the `Voice` (it can be
saved and loaded), and call `Engine::synthesize(text, &voice, &SynthOptions::default())` for every sentence. `Engine` needs
`&mut self`, so wrap it in a mutex or a worker thread if several callers share it. Requests should be queued and run one at a time.

## Tests

`cargo test` runs the text tests, which need nothing else. The synthesis and registration tests compare with the Python runtime
(ground truth in `crates/irodori-tts/tests/data`, generated by `export/12_golden_text.py` and `export/13_golden_synth.py`) and
are skipped unless `IRODORI_MODEL_DIR` and `IRODORI_TOKENIZER_DIR` are set (`IRODORI_TOL`, `IRODORI_FP16`, `IRODORI_CUDA` and
`IRODORI_ORT_DLL` are optional; use `IRODORI_FP16=1 IRODORI_TOL=0.05` for fp16 on CUDA). Current results with the fp32 graphs on CPU:

| Test | Result |
|---|---|
| Normalization, duration features, token ids (51 sentences incl. edge cases) | identical to Python |
| Synthesis (same voice, same noise) | latent relative error 4.2e-6, same frame count, same trimmed length |
| Registration (synthetic reference at 48 kHz and 44.1 kHz) | latent 1.3e-3, speaker state 5.3e-3 (see item 14) |

## Roadmap

1. Publish the converted fp16 ONNX files on Hugging Face (with the original licenses).
2. Rename the DiT graph from `dit_step2` and settle the file layout of a model release.
3. Streaming per sentence, and a stable public API before publishing the crate.

## License

MIT for the code in this repository. The models it converts have their own licenses; see [NOTICE.md](NOTICE.md).

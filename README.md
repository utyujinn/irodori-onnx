# irodori-onnx

Run [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) (v4.1 MeanFlow, Japanese TTS with zero-shot voice cloning and
emoji-based style control) **without Python or PyTorch**: convert it to ONNX, then run it from Rust on ONNX Runtime (CUDA).

> **Status: research spike.** The export pipeline works and the numbers below were measured, but the Rust runtime
> (`crates/irodori-tts`) is still an empty skeleton, and the scripts have not yet been re-verified from a clean checkout.
> Comments inside the scripts are in Japanese.

Please read [NOTICE.md](NOTICE.md) first: the model has **ethical restrictions** (no voice cloning without consent,
no deceptive deepfakes), and this project does **not** implement the SilentCipher watermark of the original runtime.

## Results so far

Model: `Aratako/Irodori-TTS-v4.1-Small-MF`, 4 sampling steps. Measured on Windows 11, RTX 4070 (12 GB), with VRChat running.

| Item | Result |
|---|---|
| Export | All 6 parts export with fully dynamic lengths: text encoder, speaker encoder, duration predictor, DiT step, codec decoder, codec encoder |
| Parity vs PyTorch (ONNX Runtime CPU) | relative L2 error 1e-6 to 1e-7 per part; end-to-end waveform correlation 1.000000 (identical noise, 173 frames, 332160 samples) |
| Parity, fp16 on CUDA | DiT step 2.3e-3; end-to-end waveform correlation 0.982 (one listener heard almost no difference) |
| Size (fp16) | 1.74 GB of ONNX (DiT 749 MB, text 636 MB, speaker 123 MB, decoder 131 MB, encoder 56 MB, duration 44 MB) |
| Speed (fp16, CUDA) | about 600 ms for 6.9 s of audio (text 71 + duration 5 + DiT 4 steps 291 + decode 228); the Python server takes about 780 ms in the same setup |
| VRAM (per process) | 1.8 GB resident (Python server: 3.4 GB). Peak while synthesizing 2.5 GB for typical sentences and 3.5 GB for 20 s of audio, **only with per-run arena shrinkage** (see below); ONNX Runtime's default behaves like the Python server (3.6 / 5.1 GB) |
| Runtime files | `onnxruntime_providers_cuda.dll` 349 MB + minimal NVIDIA runtime 1.84 GB (cuFFT is required; cuDNN `adv`/`runtime_compiled` and NVRTC are not) |

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
9. **ONNX Runtime keeps memory blocks for every distinct shape.** Pass a `RunOptions` with
   `memory.enable_memory_arena_shrinkage = "gpu:0"`: about 1.1 GB less peak VRAM, no measurable slowdown (637 ms vs 617 ms).
10. **Crate / library pairing:** the `ort` crate `2.0.0-rc.11` targets ONNX Runtime 1.23. `rc.13` targets 1.28 and does not
    load a 1.23 DLL. If your app already ships another ONNX Runtime, load this one under a different file name
    (verified: it coexists with sherpa-onnx's and VOICEVOX's copies in the same process).
11. `onnx.save(..., all_tensors_to_one_file=True)` **appends** to an existing external-data file; delete it first.

## Layout

```text
export/            Python: ONNX export, fp16 conversion, parity and memory tools
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
crates/irodori-tts/  Rust runtime (skeleton)
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
default `ref1`). Only record or use voices you have the right to use.

## Roadmap

1. Verify the scripts from a clean checkout and publish the converted fp16 ONNX files on Hugging Face (with the original licenses).
2. `crates/irodori-tts`: text normalization, tokenizer (`tokenizers` crate), duration features, MeanFlow sampler, tail trimming,
   loudness normalization and reference-voice registration, ONNX Runtime session management with the memory settings above.
3. A small CLI (text to wav) and an automatic parity check against the Python ground-truth data.

## License

MIT for the code in this repository. The models it converts have their own licenses; see [NOTICE.md](NOTICE.md).

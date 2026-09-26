# Third-party models and software

This repository contains **no model weights**. It contains scripts that convert the models below to ONNX, and (planned)
a Rust runtime that runs them. Converted weights are derivative works and must be distributed together with the
original license text and copyright notice of each model.

| Component | License | Notes |
|---|---|---|
| [Aratako/Irodori-TTS-v4.1-Small-MF](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF) | MIT + ethical restrictions | See below |
| [sbintuitions/modernbert-ja-310m](https://huggingface.co/sbintuitions/modernbert-ja-310m) | MIT | Text encoder backbone |
| [Aratako/Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim) | MIT | Audio codec (derived from facebook/dacvae, Apache-2.0) |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT | |
| NVIDIA CUDA / cuDNN runtime | NVIDIA proprietary licenses | **Not redistributed by this repository.** Check the CUDA EULA / cuDNN SLA before shipping them. |

## Ethical restrictions of Irodori-TTS (apply in addition to the MIT license)

From the model card of Irodori-TTS-v4.1-Small-MF:

- Do not use the model to clone or impersonate the voice of any individual (e.g. voice actors, celebrities, public
  figures) without their explicit consent.
- Do not use the model to generate deepfakes or synthetic speech intended to mislead others or spread misinformation.
- Users are solely responsible for ensuring their use of the generated content complies with applicable laws.

Applications built on this repository should surface these restrictions wherever reference voices can be registered.

## Watermarking

The original Python runtime applies a SilentCipher watermark to generated audio. **This repository's runtime does not
implement the watermark.** The DAC-VAE codec's built-in watermark is disabled in the Aratako fine-tune and is likewise
not part of the exported graphs. Downstream applications should decide for themselves how to disclose synthetic audio.

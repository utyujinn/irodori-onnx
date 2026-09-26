//! Irodori-TTS (v4.1 MeanFlow) inference on ONNX Runtime, without Python or PyTorch.
//!
//! Work in progress. The planned pipeline (see the repository README) is:
//! text normalization -> tokenizer -> text encoder / speaker encoder -> duration predictor
//! -> MeanFlow sampler (DiT, 4 steps) -> codec decoder -> tail trimming.

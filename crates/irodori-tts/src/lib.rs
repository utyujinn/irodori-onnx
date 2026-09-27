//! Irodori-TTS (v4.1 MeanFlow) inference on ONNX Runtime, without Python or PyTorch.
//!
//! Work in progress. The pipeline is: text normalization -> tokenizer -> text encoder -> duration predictor
//! -> MeanFlow sampler (DiT steps) -> codec decoder -> tail trimming. A reference voice is registered once with [`VoiceRegistrar`] and reused.

pub mod audio;
pub mod engine;
pub mod postprocess;
pub mod registrar;
pub mod sampler;
pub mod text;
pub mod tokenizer;
pub mod voice;

pub use engine::{Engine, EngineConfig, SynthOptions, Synthesis, SAMPLE_RATE};
pub use registrar::{Clip, VoiceRegistrar};
pub use voice::Voice;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("the text is empty after normalization")]
    EmptyText,
    #[error("model: {0}")]
    Model(String),
    #[error("audio: {0}")]
    Audio(String),
    #[error(transparent)]
    Ort(#[from] ort::Error),
    #[error(transparent)]
    Shape(#[from] ndarray::ShapeError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

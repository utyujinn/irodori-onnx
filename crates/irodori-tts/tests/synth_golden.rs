//! End-to-end check of the Rust engine against the PyTorch runtime (tests/data/synth_golden.json, produced by
//! export/13_golden_synth.py): same voice state, same text, same initial noise -> same latent.
//!
//! Needs the exported ONNX graphs, so it is skipped unless these environment variables are set:
//!   IRODORI_MODEL_DIR       folder with the ONNX graphs
//!   IRODORI_TOKENIZER_DIR   the checkpoint's tokenizer/ folder
//! Optional: IRODORI_ORT_DLL (ONNX Runtime library), IRODORI_FP16=1, IRODORI_CUDA=<device id>, IRODORI_TOL (relative L2, default 2e-4;
//! use about 3e-2 for fp16 on CUDA).

use irodori_tts::{Engine, EngineConfig, SynthOptions, Voice};
use ndarray::Array3;
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    text: String,
    steps: usize,
    seed: u64,
    speaker_state_shape: [usize; 2],
    speaker_state: Vec<f32>,
    speaker_mask: Vec<bool>,
    frames: usize,
    noise: Vec<f32>,
    z: Vec<f32>,
    samples_len: usize,
    audio_head: Vec<f32>,
}

fn config() -> Option<EngineConfig> {
    let model_dir = std::env::var("IRODORI_MODEL_DIR").ok()?;
    let tokenizer_dir = std::env::var("IRODORI_TOKENIZER_DIR").ok()?;
    Some(EngineConfig {
        model_dir: model_dir.into(),
        tokenizer_dir: tokenizer_dir.into(),
        ort_dylib: std::env::var("IRODORI_ORT_DLL").ok().map(Into::into),
        fp16: std::env::var("IRODORI_FP16").is_ok_and(|v| v == "1"),
        cuda_device: std::env::var("IRODORI_CUDA").ok().and_then(|v| v.parse().ok()),
    })
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let diff: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let norm: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (diff / norm).sqrt()
}

#[test]
fn synthesis_matches_pytorch() {
    let Some(config) = config() else {
        eprintln!("IRODORI_MODEL_DIR / IRODORI_TOKENIZER_DIR not set: skipping the synthesis test");
        return;
    };
    let tolerance: f64 = std::env::var("IRODORI_TOL").ok().and_then(|v| v.parse().ok()).unwrap_or(2e-4);
    let golden: Golden = serde_json::from_str(include_str!("data/synth_golden.json")).unwrap();

    let mut engine = Engine::new(&config).unwrap();
    let [tokens, dim] = golden.speaker_state_shape;
    let voice = Voice::from_parts(golden.speaker_state, tokens, dim, golden.speaker_mask).unwrap();
    let options = SynthOptions {
        steps: golden.steps,
        seed: golden.seed,
        noise: Some(Array3::from_shape_vec((1, golden.frames, 32), golden.noise).unwrap()),
        ..SynthOptions::default()
    };
    let out = engine.synthesize(&golden.text, &voice, &options).unwrap();

    assert_eq!(out.frames, golden.frames, "the duration predictor must give the same frame count as PyTorch");
    let latent_err = rel_l2(out.latent.as_slice().unwrap(), &golden.z);
    let head = golden.audio_head.len().min(out.samples.len());
    let audio_err = rel_l2(&out.samples[..head], &golden.audio_head[..head]);
    eprintln!("latent rel_l2 = {latent_err:.3e}, audio head rel_l2 = {audio_err:.3e}, samples {} (golden {})", out.samples.len(), golden.samples_len);
    assert!(latent_err < tolerance, "latent differs from PyTorch: {latent_err:.3e} >= {tolerance:.1e}");
    assert!(audio_err < tolerance * 10.0, "audio differs from PyTorch: {audio_err:.3e}");
    assert_eq!(out.samples.len(), golden.samples_len, "tail trimming must cut at the same sample");
}

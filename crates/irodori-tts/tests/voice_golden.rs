//! Reference voice registration against the Python runtime (tests/data/synth_golden.json, export/13_golden_synth.py).
//! The reference is a deterministic synthetic signal, so no real voice is needed. Skipped unless IRODORI_MODEL_DIR is set
//! (same environment variables as tests/synth_golden.rs).

use irodori_tts::{Clip, EngineConfig, VoiceRegistrar};
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    ref_latent: Vec<f32>,
    ref_latent_shape: [usize; 2],
    ref_latent_44k: Vec<f32>,
    ref_latent_44k_shape: [usize; 2],
    speaker_state: Vec<f32>,
    speaker_state_shape: [usize; 2],
}

/// The same formula as `synthetic_wave` in export/13_golden_synth.py.
fn synthetic_wave(sample_rate: u32) -> Vec<f32> {
    let n = (sample_rate as f64 * 2.0) as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            let tau = 2.0 * std::f64::consts::PI;
            let x = 0.3 * (tau * 220.0 * t).sin() + 0.2 * (tau * 440.0 * t * (1.0 + 0.05 * (tau * 1.5 * t).sin())).sin() + 0.1 * (tau * 1320.0 * t).sin();
            (x * 0.5 * (1.0 + (tau * 3.0 * t).sin())) as f32
        })
        .collect()
}

fn config() -> Option<EngineConfig> {
    Some(EngineConfig {
        model_dir: std::env::var("IRODORI_MODEL_DIR").ok()?.into(),
        tokenizer_dir: std::env::var("IRODORI_TOKENIZER_DIR").unwrap_or_default().into(),
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
fn registration_matches_python() {
    let Some(config) = config() else {
        eprintln!("IRODORI_MODEL_DIR not set: skipping the registration test");
        return;
    };
    let tolerance: f64 = std::env::var("IRODORI_TOL").ok().and_then(|v| v.parse().ok()).unwrap_or(5e-3);
    // The speaker encoder amplifies the small loudness difference between the ebur128 crate and audiotools (-16.575 vs -16.617 LUFS
    // for this signal, 0.04 dB), so the speaker state gets a looser bound than the latents.
    let state_tolerance = tolerance * 2.0;
    let golden: Golden = serde_json::from_str(include_str!("data/synth_golden.json")).unwrap();
    let mut registrar = VoiceRegistrar::new(&config).unwrap();

    // Same rate as the codec: only the loudness normalization can differ (ebur128 crate vs audiotools).
    let wave48 = synthetic_wave(48_000);
    let latent = registrar.encode_clip(&Clip { samples: &wave48, sample_rate: 48_000 }).unwrap();
    assert_eq!(latent.shape(), golden.ref_latent_shape);
    let err48 = rel_l2(latent.as_slice().unwrap(), &golden.ref_latent);

    // 44.1 kHz: adds the difference between the two resamplers.
    let wave44 = synthetic_wave(44_100);
    let latent44 = registrar.encode_clip(&Clip { samples: &wave44, sample_rate: 44_100 }).unwrap();
    assert_eq!(latent44.shape(), golden.ref_latent_44k_shape);
    let err44 = rel_l2(latent44.as_slice().unwrap(), &golden.ref_latent_44k);

    let voice = registrar.register(&[Clip { samples: &wave48, sample_rate: 48_000 }]).unwrap();
    assert_eq!(voice.tokens(), golden.speaker_state_shape[0]);
    let state_err = rel_l2(voice.state_slice(), &golden.speaker_state);
    eprintln!("latent rel_l2: 48k {err48:.3e}, 44.1k {err44:.3e}; speaker state {state_err:.3e}");
    assert!(err48 < tolerance, "48 kHz latent differs from Python: {err48:.3e}");
    assert!(err44 < tolerance, "44.1 kHz latent differs from Python: {err44:.3e}");
    assert!(state_err < state_tolerance, "speaker state differs from Python: {state_err:.3e}");

    let path = std::env::temp_dir().join("irodori_voice_roundtrip.irvc");
    voice.save(&path).unwrap();
    let loaded = irodori_tts::Voice::load(&path).unwrap();
    assert_eq!(loaded.state_slice(), voice.state_slice());
    let _ = std::fs::remove_file(path);
}

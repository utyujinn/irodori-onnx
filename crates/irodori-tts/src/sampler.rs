//! Pieces of the MeanFlow sampler that live outside the ONNX graphs (irodori_tts/meanflow.py, model.py).

use ndarray::Array3;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

/// Sinusoidal embedding of a scalar time (`get_timestep_embedding` in model.py): `[cos(args), sin(args)]`.
///
/// It is computed here instead of inside the DiT graph on purpose: the phase reaches about 1000 rad, which fp16 cannot
/// represent, so a graph converted to fp16 would produce garbage.
pub fn timestep_embedding(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0.0f32; dim];
    for i in 0..half {
        let freq = 1000.0 * (-(10000.0f64).ln() * i as f64 / half as f64).exp();
        let arg = t as f64 * freq;
        out[i] = arg.cos() as f32;
        out[half + i] = arg.sin() as f32;
    }
    out
}

/// Time points of the linear MeanFlow schedule, from 1.0 down to 0.0 (`steps + 1` values).
pub fn schedule(steps: usize) -> Vec<f32> {
    (0..=steps).map(|i| 1.0 - i as f32 / steps as f32).collect()
}

/// Standard normal noise of shape `(1, frames, dim)`. The stream differs from PyTorch's generator; parity tests inject the
/// Python noise instead.
pub fn gaussian_noise(seed: u64, frames: usize, dim: usize) -> Array3<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let data: Vec<f32> = (0..frames * dim).map(|_| StandardNormal.sample(&mut rng)).collect();
    Array3::from_shape_vec((1, frames, dim), data).expect("shape matches the generated length")
}

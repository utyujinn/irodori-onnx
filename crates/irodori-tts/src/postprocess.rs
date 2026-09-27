//! Post-processing of the generated latent (`find_flattening_point` in inference_runtime.py).

use ndarray::ArrayView2;

/// Frame index where the generated latent becomes flat and near zero, i.e. where the speech ends.
/// Returns `frames` (no cut) when the latent never flattens. Statistics are taken over all values of the window,
/// with the biased standard deviation, like `torch.std(unbiased=False)`.
pub fn flattening_point(latent: ArrayView2<f32>, window: usize, std_threshold: f64, mean_threshold: f64) -> usize {
    let frames = latent.nrows();
    let dim = latent.ncols();
    if frames == 0 || window == 0 {
        return frames;
    }
    let n = (window * dim) as f64;
    for start in 0..frames {
        let mut sum = 0.0f64;
        for row in start..start + window {
            if row < frames {
                sum += latent.row(row).iter().map(|&v| v as f64).sum::<f64>();
            }
        }
        let mean = sum / n;
        let mut sq = 0.0f64;
        for row in start..start + window {
            for col in 0..dim {
                let v = if row < frames { latent[[row, col]] as f64 } else { 0.0 };
                sq += (v - mean) * (v - mean);
            }
        }
        let std = (sq / n).sqrt();
        if std < std_threshold && mean.abs() < mean_threshold {
            return start;
        }
    }
    frames
}

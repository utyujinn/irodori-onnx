//! Post-processing of the generated latent (`find_flattening_point` in inference_runtime.py) and,
//! beyond what the Python runtime does, of the decoded waveform: a fade-out at the tail-trim cut
//! and leading-silence trimming (see `fade_out`/`leading_silence` below, both opt-in through
//! `SynthOptions` so the parity tests below their default configuration are unaffected).

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

/// Tapers the last `fade_samples` samples to zero (linear ramp). A hard truncation at the
/// tail-trim cut point is a waveform discontinuity, which injects energy across the whole
/// spectrum — audibly a click, or on a short utterance a brief burst of high-frequency noise
/// right where the cut lands. A few milliseconds of fade removes the discontinuity itself rather
/// than trying to detect and mask its result after the fact.
pub fn fade_out(samples: &mut [f32], fade_samples: usize) {
    let n = fade_samples.min(samples.len());
    let start = samples.len() - n;
    for (i, s) in samples[start..].iter_mut().enumerate() {
        *s *= 1.0 - (i + 1) as f32 / n as f32;
    }
}

/// Sample offset where the audio first exceeds `rms_threshold` over a `window_samples` window,
/// capped at `max_samples` as a safety net (a legitimately quiet opening — a soft consonant, a
/// trailing-off reference voice — should never be mistaken for the model's own pre-speech lead-in
/// beyond a small, fixed budget). Unlike `flattening_point`, this works on the decoded waveform
/// directly rather than the latent: it is answering "is there audible sound here", not "has the
/// model's own internal representation settled", and a simple RMS threshold on real samples is a
/// more direct match for that than reusing the latent-flatness heuristic backwards.
pub fn leading_silence(samples: &[f32], window_samples: usize, rms_threshold: f32, max_samples: usize) -> usize {
    if window_samples == 0 {
        return 0;
    }
    let limit = max_samples.min(samples.len());
    let mut start = 0;
    while start < limit {
        let end = (start + window_samples).min(samples.len());
        let window = &samples[start..end];
        let rms = (window.iter().map(|&s| s * s).sum::<f32>() / window.len() as f32).sqrt();
        if rms >= rms_threshold {
            return start;
        }
        start = end;
    }
    limit
}

/// Sample count to keep so the audio ends where it last exceeded `rms_threshold` over a
/// `window_samples` window, plus `keep_after_samples` of margin (so the cut lands past the actual
/// sound, not right against it — a natural decay/breath tail reads as part of the utterance, not
/// as the padding this is trimming). Unlike `flattening_point`, this scans the decoded waveform
/// itself rather than the latent: `flattening_point` requires the *whole* window to go flat
/// (`std`/`mean` both below their thresholds) before it will cut, and in practice reference-voice-
/// specific padding often stays just under that bar for a while longer — quiet, but not flat
/// enough to trip a stricter latent threshold without also risking a false cut into real quiet
/// speech. Returns `samples.len()` (no further cut) if the last loud window is not found, which
/// happens when nothing in the clip ever exceeds the threshold at all.
pub fn trailing_silence(samples: &[f32], window_samples: usize, rms_threshold: f32, keep_after_samples: usize) -> usize {
    if window_samples == 0 || samples.is_empty() {
        return samples.len();
    }
    let mut end = samples.len();
    while end > 0 {
        let start = end.saturating_sub(window_samples);
        let window = &samples[start..end];
        let rms = (window.iter().map(|&s| s * s).sum::<f32>() / window.len() as f32).sqrt();
        if rms >= rms_threshold {
            return (end + keep_after_samples).min(samples.len());
        }
        end = start;
    }
    samples.len()
}

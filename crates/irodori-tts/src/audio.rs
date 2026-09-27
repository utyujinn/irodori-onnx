//! Audio preparation of a reference recording, mirroring `DACVAECodec.encode_waveform` of the Python runtime:
//! resample to the codec's rate, then normalize the loudness to -16 LUFS and keep the peak at or below 1.0.

use ebur128::{EbuR128, Mode};

use crate::engine::SAMPLE_RATE;
use crate::{Error, Result};

/// Loudness the model was trained with for reference recordings.
pub const REFERENCE_LUFS: f64 = -16.0;

fn audio_error(what: &str, e: impl std::fmt::Display) -> Error {
    Error::Audio(format!("{what}: {e}"))
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Resamples mono audio to [`SAMPLE_RATE`] (48 kHz). Returns a copy when the rate already matches.
///
/// This is `torchaudio.functional.resample` with its defaults (Hann-windowed sinc, `lowpass_filter_width=6`, `rolloff=0.99`),
/// which is what the Python runtime uses. It is implemented here rather than taken from a resampling crate because the
/// references have to match the Python pipeline; a general-purpose FFT resampler produced a glitch at the start of the clip.
pub fn resample_to_codec_rate(samples: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
    if sample_rate == SAMPLE_RATE {
        return Ok(samples.to_vec());
    }
    if sample_rate == 0 || samples.is_empty() {
        return Err(Error::Audio("empty audio or a sample rate of 0".into()));
    }
    const LOWPASS_FILTER_WIDTH: f64 = 6.0;
    const ROLLOFF: f64 = 0.99;
    let g = gcd(sample_rate as usize, SAMPLE_RATE as usize);
    let (orig, new) = (sample_rate as usize / g, SAMPLE_RATE as usize / g);
    let base_freq = orig.min(new) as f64 * ROLLOFF;
    let width = (LOWPASS_FILTER_WIDTH * orig as f64 / base_freq).ceil() as usize;
    let kernel_len = 2 * width + orig;

    // One kernel per output phase: kernels[p][k] for the taps k over the input positions -width .. width + orig.
    let scale = base_freq / orig as f64;
    let kernels: Vec<Vec<f64>> = (0..new)
        .map(|p| {
            (0..kernel_len)
                .map(|k| {
                    let idx = (k as f64 - width as f64) / orig as f64;
                    let t = (-(p as f64) / new as f64 + idx) * base_freq;
                    let t = t.clamp(-LOWPASS_FILTER_WIDTH, LOWPASS_FILTER_WIDTH);
                    let window = (t * std::f64::consts::PI / LOWPASS_FILTER_WIDTH / 2.0).cos().powi(2);
                    let t = t * std::f64::consts::PI;
                    let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
                    sinc * window * scale
                })
                .collect()
        })
        .collect();

    // Zero-pad by `width` on the left and `width + orig` on the right, then output sample j (block b = j / new, phase p = j % new)
    // is the dot product of kernel p with the padded input starting at b * orig.
    let target = (samples.len() * new).div_ceil(orig);
    let mut padded = vec![0.0f64; width + samples.len() + width + orig];
    for (dst, &src) in padded[width..].iter_mut().zip(samples) {
        *dst = src as f64;
    }
    Ok((0..target)
        .map(|j| {
            let start = (j / new) * orig;
            let taps = &padded[start..start + kernel_len];
            kernels[j % new].iter().zip(taps).map(|(k, x)| k * x).sum::<f64>() as f32
        })
        .collect())
}

/// Scales the audio to [`REFERENCE_LUFS`] (ITU-R BS.1770 integrated loudness), then scales it down again if the peak exceeds 1.0.
/// Silence (no measurable loudness) is left untouched.
pub fn normalize_reference_loudness(samples: &mut [f32]) -> Result<()> {
    let mut meter = EbuR128::new(1, SAMPLE_RATE, Mode::I).map_err(|e| audio_error("loudness meter", e))?;
    meter.add_frames_f32(samples).map_err(|e| audio_error("loudness meter", e))?;
    let loudness = meter.loudness_global().map_err(|e| audio_error("loudness meter", e))?;
    if loudness.is_finite() {
        let gain = 10f64.powf((REFERENCE_LUFS - loudness) / 20.0) as f32;
        for s in samples.iter_mut() {
            *s *= gain;
        }
    }
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    if peak > 1.0 {
        for s in samples.iter_mut() {
            *s /= peak;
        }
    }
    Ok(())
}

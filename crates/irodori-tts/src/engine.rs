//! The synthesis pipeline on ONNX Runtime:
//! text -> tokens -> text encoder -> duration predictor -> MeanFlow sampler (DiT steps) -> codec decoder -> tail trim.

use std::path::{Path, PathBuf};

use ndarray::{s, Array2, Array3};
use ort::execution_providers::{cuda::ConvAlgorithmSearch, ArenaExtendStrategy, CUDAExecutionProvider};
use ort::session::{RunOptions, Session};
use ort::value::{Tensor, TensorRef};

use crate::tokenizer::TextTokenizer;
use crate::voice::Voice;
use crate::{postprocess, sampler, text, Error, Result};

/// Output sample rate of the codec decoder.
pub const SAMPLE_RATE: u32 = 48_000;
/// Audio samples per latent frame.
const HOP: usize = 1920;
const LATENT_DIM: usize = 32;
const CAPTION_DIM: usize = 512;
const TIMESTEP_DIM: usize = 512;
/// The checkpoint's default text length limit (BOS included).
const MAX_TEXT_LEN: usize = 256;
/// The checkpoint itself allows captions up to 512 tokens, but a caption here is meant to be a
/// short style instruction ("明るく元気に"), not a second sentence — capped much lower as a sanity
/// bound against a caller pasting something far longer than intended, not because longer captions
/// wouldn't work.
const MAX_CAPTION_LEN: usize = 64;
/// File stem of the DiT step graph (kept from the export scripts).
const DIT_GRAPH: &str = "dit_step2";

#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Folder with the ONNX graphs (`text_encoder.onnx`, ... or their `_fp16` variants).
    pub model_dir: PathBuf,
    /// The checkpoint's `tokenizer/` folder (`tokenizer.json`, `tokenizer_config.json`).
    pub tokenizer_dir: PathBuf,
    /// ONNX Runtime shared library to load. Give it a file name that no other ONNX Runtime in the process uses.
    /// `None` lets the `ort` crate look it up itself.
    pub ort_dylib: Option<PathBuf>,
    /// Use the `_fp16` graphs.
    pub fp16: bool,
    /// CUDA device id, or `None` to run on the CPU.
    pub cuda_device: Option<i32>,
}

#[derive(Clone, Debug)]
pub struct SynthOptions {
    /// Number of sampler steps (the MeanFlow checkpoint defaults to 4).
    pub steps: usize,
    pub seed: u64,
    pub min_seconds: f32,
    pub max_seconds: f32,
    /// Multiplies the predicted duration (above 1 speaks slower).
    pub duration_scale: f32,
    /// Cut the audio where the generated latent goes flat.
    pub trim_tail: bool,
    /// Fade the last this many milliseconds to zero at the tail-trim cut, instead of a hard
    /// truncation (which is itself an audible click/burst of noise; see postprocess::fade_out).
    /// `0.0` (the default, and what the parity tests use) disables it, matching the Python
    /// runtime's own output exactly.
    pub fade_out_ms: f32,
    /// Trim leading near-silence from the decoded audio (see postprocess::leading_silence).
    /// `false` by default, again to keep the parity tests' output identical to the Python runtime.
    pub trim_leading_silence: bool,
    /// Trim trailing near-silence the latent-based `trim_tail` left behind (see
    /// postprocess::trailing_silence). `false` by default, same reasoning.
    pub trim_trailing_silence: bool,
    /// Use this initial noise `(1, frames, 32)` instead of drawing it from `seed` (parity tests).
    pub noise: Option<Array3<f32>>,
    /// A style instruction ("明るく元気に", "落ち着いた口調で", ...) — `None`/empty skips running
    /// caption_encoder entirely and falls back to the same zero-state/all-False-mask stand-in used
    /// before this field existed (see synthesize()'s own comment on why that's a safe shortcut, not
    /// an approximation). Unlike `text`, this is NOT run through text::normalize_text: the Python
    /// training/export pipeline tokenizes captions raw (see irodori-onnx's export/common.py's
    /// real_conditions()), since a caption is a style instruction, not something ever spoken aloud.
    pub caption: Option<String>,
}

impl Default for SynthOptions {
    fn default() -> Self {
        Self {
            steps: 4,
            seed: 0,
            min_seconds: 0.5,
            max_seconds: 30.0,
            duration_scale: 1.0,
            trim_tail: true,
            fade_out_ms: 0.0,
            trim_leading_silence: false,
            trim_trailing_silence: false,
            noise: None,
            caption: None,
        }
    }
}

pub struct Synthesis {
    /// Mono audio at [`SAMPLE_RATE`].
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Number of latent frames that were generated.
    pub frames: usize,
    /// `frames` as seconds, i.e. the duration predictor's decision (after `duration_scale`,
    /// before tail-trim/fade/leading-silence trim) — the full length the model committed to
    /// generating for this text, not what ended up in `samples`. Exposed so a caller can compare
    /// it with the input text's length: an occasional bad prediction runs far longer than the
    /// text warrants and the model fills the excess with degraded audio rather than clean
    /// silence, which is otherwise indistinguishable from a normal, correctly-paced render until
    /// something actually inspects the audio or listens to it.
    pub predicted_seconds: f32,
    /// The generated latent `(1, frames, 32)` before decoding.
    pub latent: Array3<f32>,
}

pub struct Engine {
    text_encoder: Session,
    caption_encoder: Session,
    duration: Session,
    dit: Session,
    decoder: Session,
    tokenizer: TextTokenizer,
    /// Shrinks the CUDA memory arena after every run so the resident VRAM stays low.
    run_options: RunOptions,
}

fn build_session(path: &Path, cuda_device: Option<i32>) -> Result<Session> {
    let mut builder = Session::builder()?;
    if let Some(device) = cuda_device {
        // HEURISTIC: the default exhaustive search benchmarks every new input shape, which stalls the decoder for minutes.
        let cuda = CUDAExecutionProvider::default()
            .with_device_id(device)
            .with_conv_algorithm_search(ConvAlgorithmSearch::Heuristic)
            // Grow the memory arena by what is requested instead of by powers of two: with several sessions and inputs of every
            // length, power-of-two growth left about a gigabyte of VRAM unused.
            .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
            .build()
            .error_on_failure();
        builder = builder.with_execution_providers([cuda])?;
    }
    Ok(builder.commit_from_file(path)?)
}

fn to_array3(value: &ort::value::DynValue) -> Result<Array3<f32>> {
    let (shape, data) = value.try_extract_tensor::<f32>()?;
    Ok(Array3::from_shape_vec((shape[0] as usize, shape[1] as usize, shape[2] as usize), data.to_vec())?)
}

/// Number of latent frames for a predicted log1p(frames), like the Python runtime (round half to even, then clamp).
fn frames_from_prediction(log1p_frames: f32, opts: &SynthOptions) -> usize {
    let predicted = log1p_frames.exp_m1() as f64 * opts.duration_scale as f64;
    let per_second = SAMPLE_RATE as f64 / HOP as f64;
    let min_frames = ((opts.min_seconds as f64 * per_second).ceil() as usize).max(1);
    let max_frames = ((opts.max_seconds as f64 * per_second).floor() as usize).max(1);
    // `Ord::clamp` panics if min > max, which a caller-supplied `max_seconds` below
    // `min_seconds` (e.g. a per-text cap computed without accounting for a duration_scale
    // that shrinks it further) would reach; min_seconds wins since it is the sampler's own
    // floor, not something a caller-side cap is meant to override.
    let max_frames = max_frames.max(min_frames);
    (predicted.round_ties_even().max(0.0) as usize).clamp(min_frames, max_frames)
}

impl Engine {
    pub fn new(config: &EngineConfig) -> Result<Self> {
        if let Some(dylib) = &config.ort_dylib {
            ort::init_from(dylib.to_string_lossy().as_ref())?.commit();
        }
        let suffix = if config.fp16 { "_fp16" } else { "" };
        let load = |name: &str| build_session(&config.model_dir.join(format!("{name}{suffix}.onnx")), config.cuda_device);
        let mut run_options = RunOptions::new()?;
        if let Some(device) = config.cuda_device {
            run_options.add_config_entry("memory.enable_memory_arena_shrinkage", format!("gpu:{device}"))?;
        }
        Ok(Self {
            text_encoder: load("text_encoder")?,
            caption_encoder: load("caption_encoder")?,
            duration: load("duration")?,
            dit: load(DIT_GRAPH)?,
            decoder: load("codec_decoder")?,
            tokenizer: TextTokenizer::from_dir(&config.tokenizer_dir)?,
            run_options,
        })
    }

    pub fn synthesize(&mut self, text: &str, voice: &Voice, opts: &SynthOptions) -> Result<Synthesis> {
        let normalized = text::normalize_text(text);
        let normalized = normalized.trim();
        if normalized.is_empty() {
            return Err(Error::EmptyText);
        }
        let ids = self.tokenizer.encode(normalized, MAX_TEXT_LEN)?;
        let tokens = ids.len();
        let ids = Array2::from_shape_vec((1, tokens), ids.iter().map(|&i| i as i64).collect())?;
        let text_mask = Array2::from_elem((1, tokens), true);
        // A caption's mask being all-False is what actually makes it a no-op downstream (both
        // duration's masked-mean pooling and DiT's attention simply exclude masked-out caption
        // tokens), not any property of caption_state's own values — that's why the no-caption case
        // can reuse a fixed zero tensor for caption_state instead of running caption_encoder on an
        // empty string: the Python export pipeline's own empty-caption baseline confirmed this
        // (real_conditions() in irodori-onnx's export/common.py forces the mask to all-False by
        // hand rather than relying on the encoder's actual output for ""). has_caption is a second,
        // coarser gate the duration predictor alone reads (see _caption_vec in the Python model):
        // it ANDs with caption_mask, so it's really just "ignore the caption entirely" spelled out
        // as its own flag instead of relying on every mask entry happening to be False.
        let (caption_state, caption_mask, has_caption) = match opts.caption.as_deref().map(str::trim) {
            Some(caption) if !caption.is_empty() => {
                let cap_ids = self.tokenizer.encode(caption, MAX_CAPTION_LEN)?;
                let cap_tokens = cap_ids.len();
                let cap_ids = Array2::from_shape_vec((1, cap_tokens), cap_ids.iter().map(|&i| i as i64).collect())?;
                let cap_mask = Array2::from_elem((1, cap_tokens), true);
                let out = self.caption_encoder.run_with_options(
                    ort::inputs!["input_ids" => Tensor::from_array(cap_ids)?, "mask" => TensorRef::from_array_view(cap_mask.view())?],
                    &self.run_options,
                )?;
                (to_array3(&out[0])?, cap_mask, true)
            }
            _ => (Array3::<f32>::zeros((1, 1, CAPTION_DIM)), Array2::from_elem((1, 1), false), false),
        };

        let text_state = {
            let out = self.text_encoder.run_with_options(
                ort::inputs!["input_ids" => Tensor::from_array(ids)?, "mask" => TensorRef::from_array_view(text_mask.view())?],
                &self.run_options,
            )?;
            to_array3(&out[0])?
        };

        let frames = {
            let features = text::duration_features(normalized, tokens, MAX_TEXT_LEN, true);
            let out = self.duration.run_with_options(
                ort::inputs![
                    "text_state" => TensorRef::from_array_view(text_state.view())?,
                    "text_mask" => TensorRef::from_array_view(text_mask.view())?,
                    "speaker_state" => TensorRef::from_array_view(voice.state.view())?,
                    "speaker_mask" => TensorRef::from_array_view(voice.mask.view())?,
                    "duration_features" => Tensor::from_array(Array2::from_shape_vec((1, features.len()), features.to_vec())?)?,
                    "has_speaker" => Tensor::from_array(ndarray::arr1(&[true]))?,
                    "caption_state" => TensorRef::from_array_view(caption_state.view())?,
                    "caption_mask" => TensorRef::from_array_view(caption_mask.view())?,
                    "has_caption" => Tensor::from_array(ndarray::arr1(&[has_caption]))?
                ],
                &self.run_options,
            )?;
            let (_, log_frames) = out[0].try_extract_tensor::<f32>()?;
            frames_from_prediction(log_frames[0], opts)
        };

        let mut x = match &opts.noise {
            Some(noise) if noise.shape() == [1, frames, LATENT_DIM] => noise.clone(),
            Some(noise) => return Err(Error::Model(format!("injected noise has shape {:?}, expected [1, {frames}, {LATENT_DIM}]", noise.shape()))),
            None => sampler::gaussian_noise(opts.seed, frames, LATENT_DIM),
        };
        let schedule = sampler::schedule(opts.steps);
        for step in 0..opts.steps {
            let (t, next) = (schedule[step], schedule[step + 1]);
            let t_embed = Array2::from_shape_vec((1, TIMESTEP_DIM), sampler::timestep_embedding(t, TIMESTEP_DIM))?;
            let delta_embed = Array2::from_shape_vec((1, TIMESTEP_DIM), sampler::timestep_embedding(t - next, TIMESTEP_DIM))?;
            let out = self.dit.run_with_options(
                ort::inputs![
                    "x_t" => Tensor::from_array(x.clone())?,
                    "t_embed" => Tensor::from_array(t_embed)?,
                    "delta_embed" => Tensor::from_array(delta_embed)?,
                    "text_state" => TensorRef::from_array_view(text_state.view())?,
                    "text_mask" => TensorRef::from_array_view(text_mask.view())?,
                    "speaker_state" => TensorRef::from_array_view(voice.state.view())?,
                    "speaker_mask" => TensorRef::from_array_view(voice.mask.view())?,
                    "caption_state" => TensorRef::from_array_view(caption_state.view())?,
                    "caption_mask" => TensorRef::from_array_view(caption_mask.view())?
                ],
                &self.run_options,
            )?;
            let (_, velocity) = out[0].try_extract_tensor::<f32>()?;
            let dt = next - t;
            for (xi, vi) in x.iter_mut().zip(velocity) {
                *xi += vi * dt;
            }
        }

        let mut samples = {
            let out = self.decoder.run_with_options(ort::inputs!["latent" => TensorRef::from_array_view(x.view())?], &self.run_options)?;
            let (_, audio) = out[0].try_extract_tensor::<f32>()?;
            audio.to_vec()
        };
        let mut max_samples = frames * HOP;
        if opts.trim_tail {
            let flat = postprocess::flattening_point(x.slice(s![0, .., ..]), 20, 0.05, 0.1) * HOP;
            if flat > 0 {
                max_samples = max_samples.min(flat);
            }
        }
        samples.truncate(max_samples);
        if opts.trim_trailing_silence {
            // Same window/threshold as trim_leading_silence below, for the same reasons; a larger
            // margin than that one's cap, kept past the last real sound, since a breath or a
            // consonant's natural decay trailing off is still part of the utterance, not padding.
            let window = (0.01 * SAMPLE_RATE as f32) as usize;
            let margin = (0.15 * SAMPLE_RATE as f32) as usize;
            let cut = postprocess::trailing_silence(&samples, window, 0.01, margin);
            samples.truncate(cut);
        }
        if opts.fade_out_ms > 0.0 {
            postprocess::fade_out(&mut samples, (opts.fade_out_ms / 1000.0 * SAMPLE_RATE as f32) as usize);
        }
        if opts.trim_leading_silence {
            // A window short enough to not eat into a genuine soft onset, a threshold well below
            // normal speech level, and a cap far more generous than any pre-speech lead-in the
            // model has been observed to produce, so a mistuned detector fails safe (too little
            // trimmed) rather than clipping real content.
            let window = (0.01 * SAMPLE_RATE as f32) as usize;
            let cap = (0.4 * SAMPLE_RATE as f32) as usize;
            let lead = postprocess::leading_silence(&samples, window, 0.01, cap);
            samples.drain(..lead);
        }
        let predicted_seconds = (frames * HOP) as f32 / SAMPLE_RATE as f32;
        Ok(Synthesis { samples, sample_rate: SAMPLE_RATE, frames, predicted_seconds, latent: x })
    }
}

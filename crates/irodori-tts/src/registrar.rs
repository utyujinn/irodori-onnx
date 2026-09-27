//! Registering a reference voice: recordings -> codec encoder -> speaker encoder -> [`Voice`].
//!
//! This is a separate object from [`crate::Engine`] on purpose: it is only needed once per voice, so an application can
//! create it, register, and drop it to give the VRAM back.

use ndarray::{Array2, Array3};
use ort::execution_providers::{cuda::ConvAlgorithmSearch, ArenaExtendStrategy, CUDAExecutionProvider};
use ort::session::{RunOptions, Session};
use ort::value::Tensor;

use crate::audio;
use crate::engine::{EngineConfig, SAMPLE_RATE};
use crate::voice::Voice;
use crate::{Error, Result};

/// Samples per latent frame of the codec.
const HOP: usize = 1920;
const LATENT_DIM: usize = 32;
/// The checkpoint limits the reference to 120 seconds.
const MAX_REFERENCE_FRAMES: usize = 120 * SAMPLE_RATE as usize / HOP;

/// A mono recording. Mix multi-channel audio down before passing it in.
pub struct Clip<'a> {
    pub samples: &'a [f32],
    pub sample_rate: u32,
}

pub struct VoiceRegistrar {
    codec_encoder: Session,
    speaker_encoder: Session,
    run_options: RunOptions,
}

fn build_session(path: &std::path::Path, cuda_device: Option<i32>) -> Result<Session> {
    let mut builder = Session::builder()?;
    if let Some(device) = cuda_device {
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

impl VoiceRegistrar {
    pub fn new(config: &EngineConfig) -> Result<Self> {
        if let Some(dylib) = &config.ort_dylib {
            ort::init_from(dylib.to_string_lossy().as_ref())?.commit();
        }
        let suffix = if config.fp16 { "_fp16" } else { "" };
        let mut run_options = RunOptions::new()?;
        if let Some(device) = config.cuda_device {
            run_options.add_config_entry("memory.enable_memory_arena_shrinkage", format!("gpu:{device}"))?;
        }
        Ok(Self {
            // codec_encoder always loads the fp32 file, never `_fp16`, unlike every other graph
            // (Mutelink TASK.md #10): its DACVAE conv stack's fp16-on-CUDA output diverges from the
            // PyTorch reference by ~13% (vs ~0.1% on CPU with the exact same fp16 weights) — bisecting
            // it found no single bad op, just ordinary per-layer fp16 rounding differences between
            // CUDA's and CPU's Conv/Snake-activation kernels compounding additively across the ~30
            // stacked residual blocks, so there's no small block-list fix the way RMSNorm's overflow
            // had (see norm_block_list/weight_norm_block_list in irodori-onnx's export/06_fp16.py,
            // which were tried first and didn't move this number). codec_encoder only runs once per
            // voice registration, never per-synthesis, so trading its speed for fp32's correctness
            // here doesn't cost anything that matters. speaker_encoder shares fp16's suffix as before —
            // tested in isolation with the same synthetic input, it agrees with CPU to ~0.3%, the
            // registration test's own large speaker_state error traced back entirely to it being fed
            // codec_encoder's already-wrong latent, not a problem of its own.
            codec_encoder: build_session(&config.model_dir.join("codec_encoder.onnx"), config.cuda_device)?,
            speaker_encoder: build_session(&config.model_dir.join(format!("speaker_encoder{suffix}.onnx")), config.cuda_device)?,
            run_options,
        })
    }

    /// The codec latent `(frames, 32)` of one recording (resampled and loudness-normalized first).
    pub fn encode_clip(&mut self, clip: &Clip<'_>) -> Result<Array2<f32>> {
        let mut samples = audio::resample_to_codec_rate(clip.samples, clip.sample_rate)?;
        audio::normalize_reference_loudness(&mut samples)?;
        // The exported encoder expects a whole number of frames; the padding of the original model is done here (zeros on the right).
        samples.resize(samples.len().div_ceil(HOP) * HOP, 0.0);
        let wav = Array3::from_shape_vec((1, 1, samples.len()), samples)?;
        let out = self.codec_encoder.run_with_options(ort::inputs!["wav" => Tensor::from_array(wav)?], &self.run_options)?;
        let (shape, data) = out[0].try_extract_tensor::<f32>()?;
        Ok(Array2::from_shape_vec((shape[1] as usize, shape[2] as usize), data.to_vec())?)
    }

    /// Registers one voice from one or more recordings of the same speaker. Their latents are concatenated in order and cut at
    /// 120 seconds in total.
    pub fn register(&mut self, clips: &[Clip<'_>]) -> Result<Voice> {
        if clips.is_empty() {
            return Err(Error::Audio("no reference recording".into()));
        }
        let mut latent = Vec::<f32>::new();
        for clip in clips {
            latent.extend(self.encode_clip(clip)?.iter());
            if latent.len() / LATENT_DIM >= MAX_REFERENCE_FRAMES {
                break;
            }
        }
        latent.truncate(latent.len().min(MAX_REFERENCE_FRAMES * LATENT_DIM));
        let frames = latent.len() / LATENT_DIM;
        let ref_latent = Array3::from_shape_vec((1, frames, LATENT_DIM), latent)?;
        let ref_mask = Array2::from_elem((1, frames), true);
        let out = self.speaker_encoder.run_with_options(
            ort::inputs!["ref_latent" => Tensor::from_array(ref_latent)?, "ref_mask" => Tensor::from_array(ref_mask)?],
            &self.run_options,
        )?;
        let (state_shape, state) = out[0].try_extract_tensor::<f32>()?;
        let (_, mask) = out[1].try_extract_tensor::<bool>()?;
        Voice::from_parts(state.to_vec(), state_shape[1] as usize, state_shape[2] as usize, mask.to_vec())
    }
}

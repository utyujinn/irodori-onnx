//! ONNX Runtime session construction shared by [`crate::Engine`] and [`crate::VoiceRegistrar`].

use std::path::Path;

use ort::execution_providers::{cuda::ConvAlgorithmSearch, ArenaExtendStrategy, CUDAExecutionProvider};
use ort::session::Session;

use crate::Result;

/// Builds a session for the graph at `path`, on CUDA if `cuda_device` is `Some`.
///
/// `disable_memory_pattern`: graph optimization is already at its max level (Level3/"All") by
/// default (ort's own docs: "All optimizations are enabled by default") — nothing to configure
/// there. Memory pattern optimization, on the other hand, is also on by default but ort's own
/// doc for it says to turn it off "if the input size varies". [`crate::Engine`]'s sessions run
/// repeatedly for the app's whole lifetime with a different shape almost every call (text
/// length, frame count, and caption length all vary per request), so leaving it on works
/// against its own assumption rather than helping; a [`crate::VoiceRegistrar`] session, by
/// contrast, is built fresh and dropped after registering exactly one voice (one shape, one
/// call), so there is nothing for the flag to cost or save there.
pub(crate) fn build_session(path: &Path, cuda_device: Option<i32>, disable_memory_pattern: bool) -> Result<Session> {
    let mut builder = Session::builder()?;
    if disable_memory_pattern {
        builder = builder.with_memory_pattern(false)?;
    }
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

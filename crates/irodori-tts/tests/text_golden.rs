//! Compares the Rust text preprocessing with the Python runtime's output (tests/data/text_golden.json,
//! produced by export/12_golden_text.py).
//!
//! The tokenizer test needs the checkpoint's `tokenizer/` folder: set `IRODORI_TOKENIZER_DIR`, otherwise it is skipped.

use irodori_tts::text::{duration_features, normalize_text};
use irodori_tts::tokenizer::TextTokenizer;
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    max_text_len: usize,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    raw: String,
    normalized: String,
    #[serde(default)]
    empty: bool,
    #[serde(default)]
    token_ids: Vec<u32>,
    #[serde(default)]
    duration_features: Vec<f32>,
}

fn golden() -> Golden {
    serde_json::from_str(include_str!("data/text_golden.json")).unwrap()
}

#[test]
fn normalization_matches_python() {
    let mut failures = Vec::new();
    for case in golden().cases {
        // Python strips the result after normalizing.
        let got = normalize_text(&case.raw).trim().to_string();
        if got != case.normalized {
            failures.push(format!("{:?}\n   rust  : {:?}\n   python: {:?}", case.raw, got, case.normalized));
        }
    }
    assert!(failures.is_empty(), "{} mismatches:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn duration_features_match_python() {
    let golden = golden();
    let mut failures = Vec::new();
    for case in golden.cases.iter().filter(|c| !c.empty) {
        let got = duration_features(&case.normalized, case.token_ids.len(), golden.max_text_len, true);
        for (i, (g, p)) in got.iter().zip(&case.duration_features).enumerate() {
            if (g - p).abs() > 1e-6 {
                failures.push(format!("{:?} feature[{i}]: rust {g} python {p}", case.raw));
            }
        }
    }
    assert!(failures.is_empty(), "{} mismatches:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn token_ids_match_python() {
    let Ok(dir) = std::env::var("IRODORI_TOKENIZER_DIR") else {
        eprintln!("IRODORI_TOKENIZER_DIR is not set: skipping the tokenizer test");
        return;
    };
    let tokenizer = TextTokenizer::from_dir(dir).unwrap();
    let golden = golden();
    let mut failures = Vec::new();
    for case in golden.cases.iter().filter(|c| !c.empty) {
        let got = tokenizer.encode(&case.normalized, golden.max_text_len).unwrap();
        if got != case.token_ids {
            failures.push(format!("{:?}\n   rust  : {:?}\n   python: {:?}", case.normalized, got, case.token_ids));
        }
    }
    assert!(failures.is_empty(), "{} mismatches:\n{}", failures.len(), failures.join("\n"));
}

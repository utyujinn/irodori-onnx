//! Text tokenizer: the Hugging Face `tokenizer.json` of ModernBERT-ja with a manually prepended BOS token,
//! mirroring `PretrainedTextTokenizer.encode` of the Python runtime.

use std::path::Path;

use tokenizers::Tokenizer;

use crate::{Error, Result};

pub struct TextTokenizer {
    inner: Tokenizer,
    bos_id: u32,
}

impl TextTokenizer {
    /// `dir` holds `tokenizer.json` and `tokenizer_config.json` (the `tokenizer/` folder of the checkpoint).
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let inner = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("tokenizer_config.json"))?)?;
        let bos = config["bos_token"].as_str().ok_or_else(|| Error::Tokenizer("tokenizer_config.json has no bos_token".into()))?;
        let bos_id = inner.token_to_id(bos).ok_or_else(|| Error::Tokenizer(format!("BOS token {bos:?} is not in the vocabulary")))?;
        Ok(Self { inner, bos_id })
    }

    /// Token ids of an already normalized text: BOS first, then the body without any special tokens.
    /// The whole sequence is cut at `max_len` tokens like the Python runtime's truncation.
    pub fn encode(&self, text: &str, max_len: usize) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(text, false).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let mut ids = Vec::with_capacity(encoding.get_ids().len() + 1);
        ids.push(self.bos_id);
        ids.extend_from_slice(encoding.get_ids());
        ids.truncate(max_len.max(1));
        Ok(ids)
    }
}

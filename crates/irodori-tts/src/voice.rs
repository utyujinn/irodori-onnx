//! A registered voice: the speaker encoder's output for a reference recording. It is computed once per voice and
//! reused for every synthesis (the speaker encoder is not part of the per-request graphs).

use std::io::{Read, Write};
use std::path::Path;

use ndarray::{Array2, Array3};

use crate::{Error, Result};

const MAGIC: &[u8; 4] = b"IRVC";
const VERSION: u32 = 1;

#[derive(Clone, Debug)]
pub struct Voice {
    /// `(1, tokens, speaker_dim)`
    pub(crate) state: Array3<f32>,
    /// `(1, tokens)`, true where the token is valid
    pub(crate) mask: Array2<bool>,
}

impl Voice {
    pub fn from_parts(state: Vec<f32>, tokens: usize, dim: usize, mask: Vec<bool>) -> Result<Self> {
        Ok(Self { state: Array3::from_shape_vec((1, tokens, dim), state)?, mask: Array2::from_shape_vec((1, tokens), mask)? })
    }

    pub fn tokens(&self) -> usize {
        self.state.shape()[1]
    }

    /// The speaker state, row-major `(tokens, dim)`.
    pub fn state_slice(&self) -> &[f32] {
        self.state.as_slice().expect("the state is stored contiguously")
    }

    /// Small binary file: magic, version, tokens, dim, f32 little-endian state, one byte per mask entry.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(MAGIC)?;
        f.write_all(&VERSION.to_le_bytes())?;
        f.write_all(&(self.tokens() as u32).to_le_bytes())?;
        f.write_all(&(self.state.shape()[2] as u32).to_le_bytes())?;
        for v in self.state.iter() {
            f.write_all(&v.to_le_bytes())?;
        }
        f.write_all(&self.mask.iter().map(|&m| m as u8).collect::<Vec<u8>>())?;
        Ok(f.flush()?)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut head = [0u8; 16];
        f.read_exact(&mut head)?;
        if &head[0..4] != MAGIC || u32::from_le_bytes(head[4..8].try_into().unwrap()) != VERSION {
            return Err(Error::Model("not a voice file of this version".into()));
        }
        let tokens = u32::from_le_bytes(head[8..12].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(head[12..16].try_into().unwrap()) as usize;
        let mut raw = vec![0u8; tokens * dim * 4];
        f.read_exact(&mut raw)?;
        let state = raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let mut mask = vec![0u8; tokens];
        f.read_exact(&mut mask)?;
        Self::from_parts(state, tokens, dim, mask.into_iter().map(|m| m != 0).collect())
    }
}

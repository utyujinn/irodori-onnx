"""Irodori-TTS(v4.1-Small-MF)のONNX化スパイク用の共通処理(Mutelinkプロジェクト外の使い捨てスクリプト)。

- モデルをCPU/fp32でロードする(InferenceRuntime.from_keyはサーバーと同じ経路)
- ONNXは複素数演算を扱えないので、RoPEを実数版に差し替えるパッチを用意する
  (patch_real_rope()。モデル構築より前に呼ぶこと。構築時に周波数キャッシュが作られるため)
"""

import glob
import os
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8")

CKPT_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4.1-Small-MF/snapshots/*/model.safetensors"
)
# 参照声の置き場。<名前>.wav(44.1kHzモノラル)と、make_latents.pyが作る<名前>_lat.pt を置く(リポジトリには含めない)
VOICES = os.environ.get("IRODORI_VOICES_DIR", os.path.join(os.path.dirname(__file__), "voices"))
REF_NAMES = tuple(os.environ.get("IRODORI_REF_VOICES", "ref1").split(","))
REF_DIR = os.path.join(os.path.dirname(__file__), "ref")
os.makedirs(REF_DIR, exist_ok=True)


def patch_real_rope() -> None:
    import irodori_tts.model as M

    def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(end, dtype=torch.float32)
        f = torch.outer(t, freqs)
        return torch.stack([torch.cos(f), torch.sin(f)], dim=-1)  # (end, dim/2, 2): 複素数の代わりに(cos, sin)

    def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # (x0 + i*x1) * (cos + i*sin) を実数で展開したもの。元の複素数版と等価
        xr = x.float().reshape(*x.shape[:3], -1, 2)
        x0, x1 = xr[..., 0], xr[..., 1]
        cos = freqs_cis[None, :, None, :, 0]
        sin = freqs_cis[None, :, None, :, 1]
        out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
        return out.reshape_as(x).type_as(x)

    M.precompute_freqs_cis = precompute_freqs_cis
    M.apply_rotary_emb = apply_rotary_emb


def load_runtime():
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey

    ckpt = glob.glob(CKPT_GLOB)[0]
    key = RuntimeKey(checkpoint=ckpt, model_device="cpu", model_precision="fp32", codec_device="cpu", codec_precision="fp32")
    return InferenceRuntime.from_key(key)


def real_conditions(rt, text: str, ref_names=None):
    ref_names = ref_names or REF_NAMES
    """実際の文と参照声から、DiTの1ステップに渡す条件(encode_conditionsの出力)を作る。"""
    from irodori_tts.codec import patchify_latent
    from irodori_tts.text_normalization import normalize_text

    model = rt.model
    text_ids, text_mask = rt.tokenizer.batch_encode([normalize_text(text).strip()])
    lat = torch.cat([torch.load(os.path.join(VOICES, f"{n}_lat.pt"), weights_only=True) for n in ref_names], dim=0)
    ref = patchify_latent(lat.unsqueeze(0), model.cfg.latent_patch_size)
    ref_mask = torch.ones(ref.shape[:2], dtype=torch.bool)
    cap_ids, cap_mask = rt.caption_tokenizer.batch_encode([""])
    cap_mask = torch.zeros_like(cap_mask)  # 空キャプション: ランタイムと同じくマスクは全部False
    with torch.no_grad():
        enc = model.encode_conditions(
            text_input_ids=text_ids, text_mask=text_mask, ref_latent=ref, ref_mask=ref_mask,
            caption_input_ids=cap_ids, caption_mask=cap_mask,
        )
    return enc  # text_state, text_mask, ref_state, ref_mask, caption_state, caption_mask

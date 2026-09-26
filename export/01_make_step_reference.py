"""DiT 1ステップの「正解データ」を作る。--patch を付けると実数版RoPEで計算し、パッチ無しの結果と一致するか確かめる。

    python 01_make_step_reference.py            # 元の実装(複素数RoPE)で正解を保存
    python 01_make_step_reference.py --patch    # 実数版RoPEで計算し、保存済みの正解と比較
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import common

use_patch = "--patch" in sys.argv
if use_patch:
    common.patch_real_rope()
rt = common.load_runtime()
model = rt.model.eval()
print("loaded. cfg:", {k: getattr(model.cfg, k) for k in ("model_dim", "num_layers", "num_heads", "latent_dim", "text_dim", "speaker_dim")})

text_state, text_mask, spk_state, spk_mask, cap_state, cap_mask = common.real_conditions(rt, "おつかれさまです。ちょっと休憩しましょうか。")
print("conditions:", {n: tuple(t.shape) for n, t in zip(["text", "text_mask", "spk", "spk_mask", "cap", "cap_mask"], (text_state, text_mask, spk_state, spk_mask, cap_state, cap_mask))})

T = 110
g = torch.Generator().manual_seed(0)
x_t = torch.randn(1, T, model.cfg.patched_latent_dim, generator=g)
t = torch.tensor([1.0])
delta = torch.tensor([0.25])
with torch.no_grad():
    t0 = time.perf_counter()
    v = model.forward_with_encoded_conditions(
        x_t=x_t, t=t, delta_t=delta, text_state=text_state, text_mask=text_mask,
        speaker_state=spk_state, speaker_mask=spk_mask, caption_state=cap_state, caption_mask=cap_mask,
    )
print(f"step ok: out {tuple(v.shape)} in {time.perf_counter() - t0:.1f}s (CPU)")

path = os.path.join(common.REF_DIR, "dit_step_ref.pt")
if not use_patch:
    torch.save({"x_t": x_t, "t": t, "delta": delta, "text_state": text_state, "text_mask": text_mask, "spk_state": spk_state,
                "spk_mask": spk_mask, "cap_state": cap_state, "cap_mask": cap_mask, "out": v}, path)
    print("saved reference:", path)
else:
    ref = torch.load(path, weights_only=True)
    d = (v - ref["out"]).abs()
    print(f"real-RoPE vs complex-RoPE: max|diff|={d.max():.3e} rel_l2={((v - ref['out']).norm() / ref['out'].norm()):.3e}")

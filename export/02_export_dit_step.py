"""DiT 1ステップ(MeanFlowの速度予測)をONNXに書き出し、正解データと比較する(スパイク)。

条件エンコード(テキスト/参照声)は別グラフにするので、ここでは encode 済みの状態を入力に取る。
空キャプションはマスクFalseのダミー入力で渡す(ランタイムと同じ挙動)。バッチは1固定。
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import common

common.patch_real_rope()
import irodori_tts.model as _M0

_orig_tse = _M0.get_timestep_embedding  # 埋め込みの正解計算用(元の関数)
_M0.get_timestep_embedding = lambda timestep, dim: timestep  # グラフ内では何もしない(入力が既に埋め込み済み)
rt = common.load_runtime()
model = rt.model.eval()

# _rope_freqsは形が動的だとキャッシュ長の分岐がガードになるので、十分長い表を固定で使う
import irodori_tts.model as M

import irodori_tts.attention as A


def _plan(segment_masks, *, query_len, attention_dtype):
    # 元実装は int(query_len) で長さを具体値に固定してしまい、可変長のエクスポートができない。FA3を使わない経路なので中身はマスクだけでよい
    kv_mask = segment_masks[0] if len(segment_masks) == 1 else torch.cat(segment_masks, dim=1)
    return A.ContextAttentionPlan(kv_mask=kv_mask, query_len=query_len, use_fa3=False)


A.build_context_attention_plan = _plan
M.build_context_attention_plan = _plan
TABLE = M.precompute_freqs_cis(model.head_dim, 4096)
model._rope_freqs = lambda seq_len, device: TABLE[:seq_len]


class Step(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x_t, t_embed, delta_embed, text_state, text_mask, speaker_state, speaker_mask, caption_state, caption_mask):
        return self.m.forward_with_encoded_conditions(
            x_t=x_t, t=t_embed, delta_t=delta_embed, text_state=text_state, text_mask=text_mask,
            speaker_state=speaker_state, speaker_mask=speaker_mask, caption_state=caption_state, caption_mask=caption_mask,
        )


ref = torch.load(os.path.join(common.REF_DIR, "dit_step_ref.pt"), weights_only=True)
EMB = model.cfg.timestep_embed_dim
args = (ref["x_t"], _orig_tse(ref["t"], EMB), _orig_tse(ref["delta"], EMB), ref["text_state"], ref["text_mask"], ref["spk_state"], ref["spk_mask"], ref["cap_state"], ref["cap_mask"])
names = ["x_t", "t_embed", "delta_embed", "text_state", "text_mask", "speaker_state", "speaker_mask", "caption_state", "caption_mask"]

step = Step(model).eval()
with torch.no_grad():
    check = step(*args)
print("wrapper vs reference: max|diff| =", float((check - ref["out"]).abs().max()))

from torch.export import Dim

T, L, S = Dim("T", min=2, max=2048), Dim("L", min=1, max=512), Dim("S", min=2, max=1024)
# caption_state/caption_mask used to be exported as fixed shape (1, 1, 512)/(1, 1) — the only shape
# dit_step_ref.pt's reference ever had, since it comes from the empty-caption baseline
# (common.py's real_conditions() always forces an empty caption). That silently baked "caption
# sequence length is always exactly 1" into the graph, which only breaks once a real (multi-token)
# caption is actually used (TASK.md 26番) — a dynamic C dim here, separate from L (text) and S
# (speaker), lets a real caption of any length flow through the same graph the empty-caption
# shortcut already uses (see engine.rs's own comment on why the empty case still stays static).
C = Dim("C", min=1, max=512)
dynamic_shapes = {
    "x_t": {1: T}, "t_embed": None, "delta_embed": None,
    "text_state": {1: L}, "text_mask": {1: L},
    "speaker_state": {1: S}, "speaker_mask": {1: S},
    "caption_state": {1: C}, "caption_mask": {1: C},
}
out_path = os.path.join(common.REF_DIR, "dit_step2.onnx")
t0 = time.perf_counter()
with torch.no_grad():
    torch.onnx.export(step, args, out_path, input_names=names, output_names=["velocity"],
                      dynamic_shapes=dynamic_shapes, dynamo=True, external_data=True)
print(f"exported in {time.perf_counter() - t0:.1f}s ->", out_path)
total = sum(os.path.getsize(os.path.join(common.REF_DIR, f)) for f in os.listdir(common.REF_DIR) if f.startswith("dit_step2") and not f.endswith(".pt"))
print(f"onnx files total: {total / 1e6:.0f} MB")

import onnxruntime as ort

feeds = {n: a.numpy() for n, a in zip(names, args)}
for prov in (["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]):
    try:
        sess = ort.InferenceSession(out_path, providers=prov)
        if prov[0].startswith("CPU"): print("   input shapes:", {i.name: i.shape for i in sess.get_inputs() if i.name in ("x_t", "text_state", "speaker_state")})
        o = sess.run(None, feeds)[0]
        d = np.abs(o - ref["out"].numpy())
        rel = np.linalg.norm(o - ref["out"].numpy()) / np.linalg.norm(ref["out"].numpy())
        print(f"{prov[0]}: max|diff|={d.max():.3e} rel_l2={rel:.3e}")
        # 形を変えて動くか(可変長の確認): T=60, L=20
        f2 = dict(feeds)
        f2["x_t"] = np.random.randn(1, 60, 32).astype(np.float32)
        f2["text_state"] = np.random.randn(1, 20, 512).astype(np.float32)
        f2["text_mask"] = np.ones((1, 20), dtype=bool)
        print("   dynamic shape (T=60, L=20) run ok:", sess.run(None, f2)[0].shape)
    except Exception as e:  # noqa: BLE001
        print(f"{prov[0]}: FAILED {type(e).__name__}: {str(e)[:300]}")

"""fp16版のDiTグラフが、fp32版からどの演算で最初にずれるかを、中間出力の比較で特定する。"""

import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(__file__))
import common

R = common.REF_DIR
NAME = sys.argv[1] if len(sys.argv) > 1 else "dit_step2"
ref = torch.load(os.path.join(R, "dit_step_ref.pt"), weights_only=True)
import irodori_tts.model as M

emb = lambda v: M.get_timestep_embedding(v, 512)
feeds = {"x_t": ref["x_t"].numpy(), "t_embed": emb(ref["t"]).numpy(), "delta_embed": emb(ref["delta"]).numpy(), "text_state": ref["text_state"].numpy(),
         "text_mask": ref["text_mask"].numpy(), "speaker_state": ref["spk_state"].numpy(), "speaker_mask": ref["spk_mask"].numpy(),
         "caption_state": ref["cap_state"].numpy(), "caption_mask": ref["cap_mask"].numpy()}

PROV = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]


def with_outputs(path, tag, names):
    m = onnx.load(path, load_external_data=False)
    del m.graph.value_info[:]  # 変換後に古い型情報が残ることがあるので消して推論し直す
    m = onnx.shape_inference.infer_shapes(m)
    types = {v.name: v for v in list(m.graph.value_info)}
    existing = {o.name for o in m.graph.output}
    for n in names:
        if n in types and n not in existing:
            m.graph.output.append(types[n])
    out = os.path.join(R, f"_bisect_{tag}.onnx")
    onnx.save(m, out)  # 重みは元の外部データを相対パスで参照したまま
    return out, {o.name for o in m.graph.output}


m32 = onnx.load(os.path.join(R, f"{NAME}.onnx"), load_external_data=False)
nodes = m32.graph.node
# 全ノードの最初の出力を候補にする(型が推論できるものだけが実際の出力になる)
cand = [n.output[0] for n in nodes if n.output]
p32, outs32 = with_outputs(os.path.join(R, f"{NAME}.onnx"), "32", cand)
p16, outs16 = with_outputs(os.path.join(R, f"{NAME}_fp16.onnx"), "16", cand)
common_outs = [n for n in cand if n in outs32 and n in outs16]
print(f"nodes={len(nodes)} comparable intermediate tensors={len(common_outs)}", flush=True)

s32 = ort.InferenceSession(p32, providers=PROV)
s16 = ort.InferenceSession(p16, providers=PROV)
names32 = [o.name for o in s32.get_outputs()]
names16 = [o.name for o in s16.get_outputs()]
r32 = dict(zip(names32, s32.run(names32, feeds)))
r16 = dict(zip(names16, s16.run(names16, feeds)))

opof = {n.output[0]: (n.op_type, n.name) for n in nodes if n.output}
first = None
rows = []
for name in common_outs:
    a, b = r32[name], r16[name]
    if a.dtype == bool or not np.issubdtype(a.dtype, np.floating):
        continue
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = np.linalg.norm(a)
    rel = float(np.linalg.norm(a - b) / denom) if denom > 0 else float(np.linalg.norm(a - b))
    bad = bool(np.isnan(b).any() or np.isinf(b).any())
    rows.append((name, opof[name], rel, bad, float(np.abs(a).max()) if a.size else 0.0))
    if first is None and (rel > 0.05 or bad) and float(np.abs(a).max()) < 1e4:  # 3.4e38級のマスク定数は除外(fp16では大きさが変わるだけで等価)
        first = len(rows) - 1
print("最初にずれた演算(前後5件):")
lo = max(0, (first or 0) - 4)
for name, (op, nname), rel, bad, mx in rows[lo : (first or 0) + 3]:
    print(f"  {op:<18} {nname[:60]:<60} rel={rel:.2e} nan/inf={bad} absmax(fp32)={mx:.3g}")
big = sorted(rows, key=lambda r: -r[4])[:6]
print("fp32側で絶対値が大きい中間テンソル(fp16の上限は65504):")
for name, (op, nname), rel, bad, mx in big:
    print(f"  {op:<18} {nname[:60]:<60} absmax={mx:.3g} rel={rel:.1e}")
for f in (p32, p16):
    os.remove(f)

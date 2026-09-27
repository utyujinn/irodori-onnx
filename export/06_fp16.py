"""ONNXの各グラフをfp16に変換する(入出力はfp32のまま)。サイズの削減と、後段の一致度/速度確認用のスパイク。

    python 06_fp16.py                # 全部
    python 06_fp16.py dit_step ...   # 指定した部品だけ
"""

import os
import sys
import time

from collections import defaultdict

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16

def norm_block_list(model):
    """RMSNormの二乗平均(x*x -> ReduceMean -> eps加算 -> Sqrt -> 逆数)をfp32のまま残すノード名を返す。
    活性値が大きい層ではx*xがfp16の上限(65504)を超えてinfになるため。"""
    nodes = list(model.graph.node)
    producer = {o: n for n in nodes for o in n.output}
    consumers = defaultdict(list)
    for n in nodes:
        for i in n.input:
            consumers[i].append(n)
    block = set()
    for n in nodes:
        if n.op_type != "ReduceMean":
            continue
        block.add(n.name)
        pr = producer.get(n.input[0])
        if pr is not None and pr.op_type in ("Mul", "Pow"):
            block.add(pr.name)
        frontier = [n]
        for _ in range(4):
            nxt = []
            for f in frontier:
                for o in f.output:
                    for c in consumers[o]:
                        if c.op_type in ("Add", "Sqrt", "Reciprocal", "Div"):
                            block.add(c.name)
                            nxt.append(c)
            frontier = nxt
    return sorted(block)


def drop_roundtrip_casts(m):
    """fp32のまま残したノードの前後には Cast(fp16) -> Cast(fp32) の往復が入り、連続するfp32ノード間で値がfp16の範囲に丸められて
    しまう(7e4のような値がinfになる)。往復Castを取り除いて、fp32のまま次のfp32ノードへ渡す。"""
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}

    def cast_to(n):
        return next((a.i for a in n.attribute if a.name == "to"), None)

    rename, dead = {}, set()
    for n in g.node:
        if n.op_type == "Cast" and cast_to(n) == 1:
            p = prod.get(n.input[0])
            if p is not None and p.op_type == "Cast" and cast_to(p) == 10:
                rename[n.output[0]] = p.input[0]
                dead.add(n.name)
    for n in g.node:
        for i, name in enumerate(n.input):
            while name in rename:
                name = rename[name]
            n.input[i] = name
    keep = [n for n in g.node if n.name not in dead]
    used = {i for n in keep for i in n.input} | {o.name for o in g.output}
    # 出力が誰にも使われなくなったfp16化Castも消す
    keep = [n for n in keep if not (n.op_type == "Cast" and cast_to(n) == 10 and n.output[0] not in used)]
    removed = len(g.node) - len(keep)
    del g.node[:]
    g.node.extend(keep)
    return removed


R = os.path.join(os.path.dirname(__file__), "ref")
# codec_encoder is deliberately NOT in this default list: its fp16 output on CUDA diverges from the
# PyTorch reference by ~13% (vs ~0.1% on CPU with the exact same weights) — bisecting found no single
# bad op to block-list the way norm_block_list handles RMSNorm's overflow, just ordinary per-layer
# fp16 rounding differences between CUDA's and CPU's Conv/Snake-activation kernels compounding
# additively across ~30 stacked residual blocks in its DACVAE conv stack. registrar.rs (irodori-tts)
# always loads the plain fp32 codec_encoder.onnx instead — it only runs once per voice registration,
# never per-synthesis, so trading its speed for fp32's correctness costs nothing that matters (see
# Mutelink TASK.md #10 for the full investigation). Pass it explicitly on the command line if you
# specifically want an (unused-by-anything) fp16 copy for comparison.
names = sys.argv[1:] or ["text_encoder", "speaker_encoder", "duration", "dit_step", "codec_decoder"]
for n in names:
    src = os.path.join(R, f"{n}.onnx")
    dst = os.path.join(R, f"{n}_fp16.onnx")
    t0 = time.perf_counter()
    m = onnx.load(src)  # 外部データも読み込む
    blk = norm_block_list(m)
    m16 = convert_float_to_float16(m, keep_io_types=True, node_block_list=blk)
    removed = drop_roundtrip_casts(m16)
    for f in (dst, dst + ".data"):  # onnx.saveは既存の外部データファイルに追記してしまうので先に消す
        if os.path.exists(f):
            os.remove(f)
    onnx.save(m16, dst, save_as_external_data=True, all_tensors_to_one_file=True, location=f"{n}_fp16.onnx.data", size_threshold=1024)
    size = sum(os.path.getsize(os.path.join(R, f)) for f in os.listdir(R) if f.startswith(f"{n}_fp16.onnx"))
    print(f"{n}: fp16 変換 {time.perf_counter() - t0:.0f}s -> {size / 1e6:.0f}MB (fp32のまま残した正規化ノード: {len(blk)}, 往復Castを{removed}個削除)", flush=True)

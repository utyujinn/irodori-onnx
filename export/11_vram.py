"""ONNX版(fp16)の実行時VRAMと速度を、プロセス単位の専用VRAMカウンタで測る。

    python 11_vram.py [shrink] [long]
      shrink: 1回の実行ごとにONNX Runtimeのメモリアリーナを縮小する(実行中のピークもバックグラウンドで測る)
      long  : 音声20秒級の長い入力(T=520)も混ぜる

実行時に必要な4グラフ(テキスト/長さ予測は省略、DiT/デコーダ/テキスト)で、長さの違う入力を30回流す。
参照声エンコーダとコーデックのエンコーダは登録時だけなので含めない。
"""

import os
import subprocess
import sys
import threading
import time

import numpy as np
import onnxruntime as ort

R = os.path.join(os.path.dirname(__file__), "ref")
PID = os.getpid()
FLAGS = set(sys.argv[1:])
PROV = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC", "arena_extend_strategy": "kSameAsRequested"}), "CPUExecutionProvider"]
RUN = ort.RunOptions()
if "shrink" in FLAGS:
    RUN.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")


def vram_mb():
    cmd = f"((Get-Counter '\\GPU Process Memory(pid_{PID}_*)\\Dedicated Usage').CounterSamples | Measure-Object CookedValue -Sum).Sum"
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.strip()
    return float(out) / 1048576 if out else float("nan")


S = {}
for n in ("text_encoder_fp16", "dit_step2_fp16", "codec_decoder_fp16"):
    S[n] = ort.InferenceSession(os.path.join(R, f"{n}.onnx"), providers=PROV)
loaded = vram_mb()

peak = [loaded]
stop = threading.Event()


def sampler():
    while not stop.is_set():
        peak[0] = max(peak[0], vram_mb())


th = threading.Thread(target=sampler, daemon=True)
th.start()

rng = np.random.default_rng(0)
emb = lambda: rng.standard_normal((1, 512)).astype(np.float32)
SHAPES = [(8, 60), (12, 110), (20, 173), (30, 260), (10, 90), (25, 200), (15, 130), (18, 150), (22, 190), (9, 70)]
if "long" in FLAGS:
    SHAPES = SHAPES[:5] + [(60, 520)] + SHAPES[5:]
spk = rng.standard_normal((1, 149, 768)).astype(np.float32)
times = []
for L, T in SHAPES * 3:
    t0 = time.perf_counter()
    S["text_encoder_fp16"].run(None, {"input_ids": rng.integers(1, 1000, (1, L)).astype(np.int64), "mask": np.ones((1, L), dtype=bool)}, RUN)
    ts = rng.standard_normal((1, L, 512)).astype(np.float32)
    x = rng.standard_normal((1, T, 32)).astype(np.float32)
    for _ in range(4):
        x = x + S["dit_step2_fp16"].run(None, {"x_t": x, "t_embed": emb(), "delta_embed": emb(), "text_state": ts, "text_mask": np.ones((1, L), dtype=bool),
                                              "speaker_state": spk, "speaker_mask": np.ones((1, 149), dtype=bool),
                                              "caption_state": np.zeros((1, 1, 512), dtype=np.float32), "caption_mask": np.zeros((1, 1), dtype=bool)}, RUN)[0] * 0.01
    S["codec_decoder_fp16"].run(None, {"latent": x}, RUN)
    times.append((time.perf_counter() - t0) * 1000)
stop.set()
th.join()
print(f"[{' '.join(sorted(FLAGS)) or 'default'}] ロード直後 {loaded:.0f} MB | 実行中ピーク {peak[0]:.0f} MB | 1回あたり(中央値) {np.median(times):.0f} ms")

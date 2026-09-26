"""NVIDIAランタイムDLLを絞った状態で、fp16の3グラフ(テキスト/DiT/デコーダ)がCUDAで動くかを確かめる。
呼び出し側でPATHを絞り、torchをimportしない別プロセスとして実行すること。

    python 10_min_cuda.py <DLLを置いたディレクトリ>
"""

import os
import sys

dll_dir = sys.argv[1]
os.environ["PATH"] = dll_dir + os.pathsep + os.environ["SystemRoot"] + "\\System32"
os.add_dll_directory(dll_dir)

import numpy as np
import onnxruntime as ort

R = os.path.join(os.path.dirname(__file__), "ref")
PROV = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]


def make(name):
    s = ort.InferenceSession(os.path.join(R, f"{name}.onnx"), providers=PROV)
    if "CUDAExecutionProvider" not in s.get_providers():
        raise RuntimeError(f"{name}: CUDAプロバイダが使われていない(DLL不足でCPUにフォールバック): {s.get_providers()}")
    return s


rng = np.random.default_rng(0)
try:
    t = make("text_encoder_fp16")
    t.run(None, {"input_ids": rng.integers(1, 1000, (1, 12)).astype(np.int64), "mask": np.ones((1, 12), dtype=bool)})
    d = make("dit_step2_fp16")
    d.run(None, {"x_t": rng.standard_normal((1, 120, 32)).astype(np.float32), "t_embed": rng.standard_normal((1, 512)).astype(np.float32),
                 "delta_embed": rng.standard_normal((1, 512)).astype(np.float32), "text_state": rng.standard_normal((1, 12, 512)).astype(np.float32),
                 "text_mask": np.ones((1, 12), dtype=bool), "speaker_state": rng.standard_normal((1, 149, 768)).astype(np.float32),
                 "speaker_mask": np.ones((1, 149), dtype=bool), "caption_state": np.zeros((1, 1, 512), dtype=np.float32), "caption_mask": np.zeros((1, 1), dtype=bool)})
    c = make("codec_decoder_fp16")
    c.run(None, {"latent": rng.standard_normal((1, 120, 32)).astype(np.float32)})
    print("OK: 3グラフともCUDAで実行できた")
except Exception as e:  # noqa: BLE001
    print("NG:", type(e).__name__, str(e)[:200].replace("\n", " "))

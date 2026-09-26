"""コーデックのエンコーダ(正規化済み48kHzモノラル波形 -> latent)をONNXに書き出す(参照声の登録用スパイク)。

ラウドネス正規化とリサンプルはグラフの外(Rust側)で行う前提。deterministic_encode の経路
(encoder -> quantizer.in_proj -> 前半をmeanとして採用)をそのまま再現する。
比較対象は、make_latents.py で保存済みの <参照声名>_lat.pt(同じ前処理をPythonでかけたもの)。
"""

import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(__file__))
import common

rt = common.load_runtime()
codec = rt.codec
print("codec sr:", codec.sample_rate, "hop:", getattr(codec.model, "hop_length", None))


class CodecEncode(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.m = c.model

    def forward(self, wav):  # (1,1,samples) 正規化済み
        z = self.m.encoder(self.m._pad(wav))
        mean, _ = self.m.quantizer.in_proj(z).chunk(2, dim=1)
        return mean.transpose(1, 2).contiguous()  # (1,T,32)


data, sr = sf.read(os.path.join(common.VOICES, f"{common.REF_NAMES[0]}.wav"), dtype="float32")
wav = torch.from_numpy(data).view(1, 1, -1)
wav = torchaudio.functional.resample(wav, sr, codec.sample_rate)
norm = codec._normalize_loudness(wav[0, 0], codec.sample_rate, -16.0).view(1, 1, -1)
expected = torch.load(os.path.join(common.VOICES, f"{common.REF_NAMES[0]}_lat.pt"), weights_only=True)
with torch.no_grad():
    got = CodecEncode(codec)(norm)[0]
print("python encoder vs saved latent:", tuple(got.shape), tuple(expected.shape), "max|diff|", float((got - expected).abs().max()))

from torch.export import Dim

path = os.path.join(common.REF_DIR, "codec_encoder.onnx")
t0 = time.perf_counter()
try:
    with torch.no_grad():
        torch.onnx.export(CodecEncode(codec).eval(), (norm,), path, input_names=["wav"], output_names=["latent"],
                          dynamic_shapes={"wav": {2: Dim("Ns", min=4800, max=48000 * 130)}}, dynamo=True, external_data=True)
    print(f"exported in {time.perf_counter() - t0:.1f}s")
    import onnxruntime as ort

    for prov in (["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]):
        sess = ort.InferenceSession(path, providers=prov)
        o = sess.run(None, {"wav": norm.numpy()})[0][0]
        print(f"{prov[0]}: rel_l2 vs saved latent = {np.linalg.norm(o - expected.numpy()) / np.linalg.norm(expected.numpy()):.2e}")
        # 別の長さでも動くか
        w2 = np.random.randn(1, 1, 48000 * 3 + 123).astype(np.float32) * 0.05
        print("   other length ok:", sess.run(None, {"wav": w2})[0].shape)
except Exception as e:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    print("EXPORT FAILED:", type(e).__name__, str(e)[:300])

"""参照wavを、Irodori-TTS-Serverと同じ条件(-16dB正規化+ピーク保護)でコーデックにかけ、latentとして保存する。
サーバーが毎回やっている参照音声のエンコード(prepare_reference)を事前計算に置き換えるためのもの。CPUで実行(VRAMを使わない)。"""
import os
import sys
import time

import torch
import soundfile as sf
from irodori_tts.codec import DACVAECodec

sys.stdout.reconfigure(encoding="utf-8")
VOICES = os.environ.get("IRODORI_VOICES_DIR", os.path.join(os.path.dirname(__file__), "voices"))
codec = DACVAECodec.load(repo_id="Aratako/Semantic-DACVAE-Japanese-32dim", device="cpu", dtype=torch.float32)
for name in os.environ.get("IRODORI_REF_VOICES", "ref1").split(","):
    data, sr = sf.read(rf"{VOICES}\{name}.wav", dtype="float32")
    wav = torch.from_numpy(data).unsqueeze(0)  # (1, T) モノラル
    t = time.perf_counter()
    lat = codec.encode_waveform(wav.unsqueeze(0), sample_rate=int(sr), normalize_db=-16.0, ensure_max=True).cpu()
    lat = lat.squeeze(0).contiguous()  # (T_latent, D_latent)
    torch.save(lat, rf"{VOICES}\{name}_lat.pt")
    print(name, tuple(lat.shape), f"{time.perf_counter() - t:.1f}s on CPU")

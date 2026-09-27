"""合成パイプラインの正解データを作る(Rust側の一致確認用)。実在の声は使わず、決定的な合成波形から「声」を作る。

- 参照波形: 数式で作った2秒の48kHz信号(Rust側テストでも同じ式で生成する)
- 参照声の条件: コーデックのエンコード(-16dB正規化、ピーク保護)-> 話者エンコーダ(PyTorch)
- 合成: 固定シードのノイズ + PyTorchのMeanFlowサンプラーで最終latentを作る
出力: crates/irodori-tts/tests/data/synth_golden.json
"""

import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import common

TEXT = "おはよう。"
STEPS, SEED = 4, 1234
SR = 48000


def synthetic_wave(sr: int = SR, seconds: float = 2.0) -> np.ndarray:
    """Rust側(tests/synth_golden.rs)と同じ式。"""
    t = np.arange(int(sr * seconds), dtype=np.float64) / sr
    x = 0.3 * np.sin(2 * math.pi * 220 * t) + 0.2 * np.sin(2 * math.pi * 440 * t * (1 + 0.05 * np.sin(2 * math.pi * 1.5 * t))) + 0.1 * np.sin(2 * math.pi * 1320 * t)
    x *= 0.5 * (1 + np.sin(2 * math.pi * 3 * t))
    return x.astype(np.float32)


rt = common.load_runtime()
model, codec = rt.model, rt.codec
HOP = int(codec.model.hop_length)
from irodori_tts.codec import patchify_latent
from irodori_tts.duration import build_duration_features
from irodori_tts.inference_runtime import find_flattening_point
from irodori_tts.meanflow import sample_euler_meanflow
from irodori_tts.rf import _make_rng
from irodori_tts.text_normalization import normalize_text

wave = synthetic_wave()
with torch.inference_mode():
    lat = codec.encode_waveform(torch.from_numpy(wave)[None, None], sample_rate=SR, normalize_db=-16.0, ensure_max=True)  # (1,T,32)
    # リサンプルが必要な経路(44.1kHz入力)の正解。torchaudioのリサンプルとRustのリサンプルの差を見る
    lat44 = codec.encode_waveform(torch.from_numpy(synthetic_wave(44100))[None, None], sample_rate=44100, normalize_db=-16.0, ensure_max=True)
    ref_latent = patchify_latent(lat, model.cfg.latent_patch_size)
    ref_mask = torch.ones(ref_latent.shape[:2], dtype=torch.bool)
    norm = normalize_text(TEXT).strip()
    ids, mask = rt.tokenizer.batch_encode([norm], max_length=rt.default_text_max_len)
    cap_ids, cap_mask = rt.caption_tokenizer.batch_encode([""])
    cap_mask = torch.zeros_like(cap_mask)
    enc = model.encode_conditions(text_input_ids=ids, text_mask=mask, ref_latent=ref_latent, ref_mask=ref_mask, caption_input_ids=cap_ids, caption_mask=cap_mask)
    feat = build_duration_features([norm], token_counts=mask.sum(dim=1), max_text_len=rt.default_text_max_len, has_speaker=[True])
    pred_log = model.predict_duration_log_frames(text_state=enc[0], text_mask=enc[1], speaker_state=enc[2], speaker_mask=enc[3], duration_features=feat,
                                                 has_speaker=torch.tensor([True]), caption_state=enc[4], caption_mask=enc[5], has_caption=torch.tensor([False]))
    pred_frames = float(torch.expm1(pred_log).float().mean())
    frames = max(max(1, math.ceil(0.5 * codec.sample_rate / HOP)), min(math.floor(30.0 * codec.sample_rate / HOP), int(round(pred_frames))))
    rng, _ = _make_rng(seed=SEED, device=torch.device("cpu"))
    noise = torch.randn((1, frames, model.cfg.patched_latent_dim), generator=rng, dtype=torch.float32)
    z = sample_euler_meanflow(model=model, text_input_ids=ids, text_mask=mask, ref_latent=ref_latent, ref_mask=ref_mask, sequence_length=frames,
                              caption_input_ids=cap_ids, caption_mask=cap_mask, num_steps=STEPS, seed=SEED)[:, :frames]
    fp = find_flattening_point(z[0], window_size=20, std_threshold=0.05, mean_threshold=0.1)
    audio = codec.decode_latent(z).cpu()[0, 0]
    samples = min(frames * HOP, fp * HOP if fp * HOP > 0 else 10**12)


def arr(t):
    return t.detach().float().cpu().numpy().reshape(-1).tolist()


out = {
    "text": TEXT, "steps": STEPS, "seed": SEED, "sample_rate": SR, "hop": HOP,
    "ref_latent_shape": list(lat.shape[1:]), "ref_latent": arr(lat),
    "ref_latent_44k_shape": list(lat44.shape[1:]), "ref_latent_44k": arr(lat44),
    "speaker_state_shape": list(enc[2].shape[1:]), "speaker_state": arr(enc[2]), "speaker_mask": enc[3][0].tolist(),
    "pred_frames": pred_frames, "frames": frames, "noise": arr(noise), "z": arr(z),
    "flatten_point": int(fp), "samples_len": int(samples),
    "audio_head": audio[:4000].tolist(),
}
dst = os.path.join(os.path.dirname(__file__), "..", "crates", "irodori-tts", "tests", "data", "synth_golden.json")
with open(dst, "w") as f:
    json.dump(out, f, separators=(",", ":"))
print(f"frames={frames} pred_frames={pred_frames:.2f} flatten={fp} samples={samples} ref_latent={tuple(lat.shape)} speaker_state={tuple(enc[2].shape)} -> {os.path.getsize(dst) / 1e3:.0f} KB")

"""ONNXの6部品だけで合成し、元のPyTorch実装と同じノイズで比較する(エンドツーエンドの一致確認 + 段階別の速度)。

    python 05_e2e_onnx.py [cpu|cuda] [文]

Rustに移植する処理の手順書も兼ねる: 正規化 -> トークナイズ -> text_encoder / speaker_encoder -> duration
-> ノイズ -> DiT x steps(MeanFlow) -> 切り詰め -> codec_decoder -> 末尾トリム。ウォーターマークは移植しない方針なので含めない。
"""

import math
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(__file__))
import common

MODE = sys.argv[1] if len(sys.argv) > 1 else "cuda"
TEXT = sys.argv[2] if len(sys.argv) > 2 else "おつかれさまです。ちょっと休憩しましょうか。"
STEPS, SEED = 4, 1234
# ORTのCUDAプロバイダは既定で畳み込みのアルゴリズムを入力の形ごとに総当たり探索する(cuDNNの初回ペナルティと同種の問題)。ヒューリスティックにする
PROV = ["CPUExecutionProvider"] if MODE == "cpu" else [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC", "arena_extend_strategy": "kSameAsRequested"}), "CPUExecutionProvider"]

rt = common.load_runtime()  # 元の実装のまま(パッチなし)= 正解側
model, codec = rt.model, rt.codec
HOP = int(codec.model.hop_length)
from irodori_tts.codec import patchify_latent
from irodori_tts.duration import build_duration_features
from irodori_tts.inference_runtime import find_flattening_point
from irodori_tts.meanflow import sample_euler_meanflow
from irodori_tts.rf import _make_rng
from irodori_tts.text_normalization import normalize_text

norm_text = normalize_text(TEXT).strip()
lat = torch.cat([torch.load(os.path.join(common.VOICES, f"{n}_lat.pt"), weights_only=True) for n in common.REF_NAMES], dim=0)
ref_latent = patchify_latent(lat.unsqueeze(0), model.cfg.latent_patch_size)
ref_mask = torch.ones(ref_latent.shape[:2], dtype=torch.bool)
cap_ids, cap_mask = rt.caption_tokenizer.batch_encode([""])
cap_mask = torch.zeros_like(cap_mask)
TEXT_MAX_LEN = rt.default_text_max_len
print(f"text: {norm_text!r}  text_max_len={TEXT_MAX_LEN}  hop={HOP}  providers={MODE}", flush=True)

# ================= 正解側(PyTorch、ランタイムと同じ手順) =================
with torch.inference_mode():
    ids_p, mask_p = rt.tokenizer.batch_encode([norm_text], max_length=TEXT_MAX_LEN)  # ランタイムは256にパディングする
    enc = model.encode_conditions(text_input_ids=ids_p, text_mask=mask_p, ref_latent=ref_latent, ref_mask=ref_mask,
                                  caption_input_ids=cap_ids, caption_mask=cap_mask)
    feat = build_duration_features([norm_text], token_counts=mask_p.sum(dim=1), max_text_len=TEXT_MAX_LEN, has_speaker=[True])
    pred_log = model.predict_duration_log_frames(text_state=enc[0], text_mask=enc[1], speaker_state=enc[2], speaker_mask=enc[3],
                                                 duration_features=feat, has_speaker=torch.tensor([True]),
                                                 caption_state=enc[4], caption_mask=enc[5], has_caption=torch.tensor([False]))
    frames_t = int(round(float(torch.expm1(pred_log).float().mean())))
    frames_t = max(max(1, math.ceil(0.5 * codec.sample_rate / HOP)), min(math.floor(30.0 * codec.sample_rate / HOP), frames_t))
    z_t = sample_euler_meanflow(model=model, text_input_ids=ids_p, text_mask=mask_p, ref_latent=ref_latent, ref_mask=ref_mask,
                                sequence_length=frames_t, caption_input_ids=cap_ids, caption_mask=cap_mask, num_steps=STEPS, seed=SEED)
    z_t = z_t[:, :frames_t]
    wav_t = codec.decode_latent(z_t).cpu()[0, 0].numpy()
    fp = find_flattening_point(z_t[0], window_size=20, std_threshold=0.05, mean_threshold=0.1)
    wav_t = wav_t[: min(frames_t * HOP, fp * HOP if fp * HOP > 0 else 10**12)]
print(f"[torch ] frames={frames_t} flatten_point={fp} samples={wav_t.shape[0]} ({wav_t.shape[0] / codec.sample_rate:.2f}s)")

# ================= ONNX側 =================
R = common.REF_DIR
opts = ort.SessionOptions()
S = {}
for n in ("text_encoder", "speaker_encoder", "duration", "dit_step2", "codec_decoder"):
    t0 = time.perf_counter()
    S[n] = ort.InferenceSession(os.path.join(R, f"{n}{os.environ.get('ONNX_SUFFIX', '')}.onnx"), sess_options=opts, providers=PROV)
    print(f"session {n} loaded in {time.perf_counter() - t0:.1f}s", flush=True)
CAP_STATE = np.zeros((1, 1, 512), dtype=np.float32) if float(enc[4].abs().max()) == 0.0 else enc[4].numpy()
print("caption state is all zeros:", float(enc[4].abs().max()) == 0.0)
CAP_MASK = np.zeros((1, 1), dtype=bool)
schedule = np.linspace(1.0, 0.0, STEPS + 1, dtype=np.float32)


def timestep_embedding(t, dim=512):
    """Irodoriの正弦波時刻埋め込み(model.get_timestep_embeddingと同じ式)。fp16グラフの精度問題を避けるためグラフの外で計算する。"""
    half = dim // 2
    freqs = 1000.0 * np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float32) / half)
    args = np.float32(t) * freqs
    return np.concatenate([np.cos(args), np.sin(args)])[None, :].astype(np.float32)


def flattening_point(latent, window=20, std_thr=0.05, mean_thr=0.1):
    total = latent.shape[0]
    padded = np.concatenate([latent, np.zeros((window, latent.shape[1]), dtype=latent.dtype)], axis=0)
    for i in range(padded.shape[0] - window):
        w = padded[i : i + window]
        if w.std() < std_thr and abs(w.mean()) < mean_thr:  # torch: std(unbiased=False), mean over all elements
            return i
    return total


def synth_onnx(noise=None, timings=None):
    tm = timings if timings is not None else {}
    t0 = time.perf_counter()
    ids, mask = rt.tokenizer.batch_encode([norm_text])  # 実際の長さのまま(パディングしない)
    ids_np, mask_np = ids.numpy(), mask.numpy()
    text_state = S["text_encoder"].run(None, {"input_ids": ids_np, "mask": mask_np})[0]
    tm["text"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    spk_state, spk_mask = S["speaker_encoder"].run(None, {"ref_latent": ref_latent.numpy(), "ref_mask": ref_mask.numpy()})
    tm["speaker"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    feats = build_duration_features([norm_text], token_counts=[int(mask_np.sum())], max_text_len=TEXT_MAX_LEN, has_speaker=[True]).numpy()
    log_frames = S["duration"].run(None, {"text_state": text_state, "text_mask": mask_np, "speaker_state": spk_state, "speaker_mask": spk_mask,
                                          "duration_features": feats, "has_speaker": np.array([True]), "caption_state": CAP_STATE,
                                          "caption_mask": CAP_MASK, "has_caption": np.array([False])})[0]
    frames = int(round(float(np.expm1(log_frames).mean())))
    frames = max(max(1, math.ceil(0.5 * 48000 / HOP)), min(math.floor(30.0 * 48000 / HOP), frames))
    tm["duration"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if noise is None:
        rng, _ = _make_rng(seed=SEED, device=torch.device("cpu"))
        noise = torch.randn((1, frames, model.cfg.patched_latent_dim), generator=rng, dtype=torch.float32).numpy()
    x = noise
    for i in range(STEPS):
        tv, nv = schedule[i], schedule[i + 1]
        v = S["dit_step2"].run(None, {"x_t": x, "t_embed": timestep_embedding(tv), "delta_embed": timestep_embedding(tv - nv),
                                     "text_state": text_state, "text_mask": mask_np, "speaker_state": spk_state, "speaker_mask": spk_mask,
                                     "caption_state": CAP_STATE, "caption_mask": CAP_MASK})[0]
        x = x + v * (nv - tv)
    tm["dit"] = time.perf_counter() - t0
    z = x[:, :frames]
    t0 = time.perf_counter()
    wav = S["codec_decoder"].run(None, {"latent": z})[0][0, 0]
    fpn = flattening_point(z[0])
    wav = wav[: min(frames * HOP, fpn * HOP if fpn * HOP > 0 else 10**12)]
    tm["decode"] = time.perf_counter() - t0
    return wav, z, frames, fpn


timings = {}
wav_o, z_o, frames_o, fp_o = synth_onnx(timings=timings)
print("first run stage times (s):", {k: round(v, 2) for k, v in timings.items()}, flush=True)
print(f"[onnx  ] frames={frames_o} flatten_point={fp_o} samples={wav_o.shape[0]} ({wav_o.shape[0] / 48000:.2f}s)")
n = min(len(wav_o), len(wav_t))
corr = float(np.corrcoef(wav_o[:n], wav_t[:n])[0, 1])
rel = float(np.linalg.norm(wav_o[:n] - wav_t[:n]) / np.linalg.norm(wav_t[:n]))
lat_rel = float(np.linalg.norm(z_o - z_t.numpy()) / np.linalg.norm(z_t.numpy())) if z_o.shape == tuple(z_t.shape) else float("nan")
print(f"[parity] latent rel_l2={lat_rel:.3e}  waveform corr={corr:.6f}  rel_l2={rel:.3e}  len torch/onnx={len(wav_t)}/{len(wav_o)}")
import soundfile as sf

sf.write(os.path.join(R, f"e2e_{MODE}{os.environ.get('ONNX_SUFFIX', '')}_onnx.wav"), wav_o, 48000)
sf.write(os.path.join(R, f"e2e_{MODE}_torch.wav"), wav_t, 48000)

# ---- 速度(ウォームアップ後、同じ文を新しいノイズで5回)
acc = {}
for k in range(6):
    tm = {}
    rng = torch.Generator().manual_seed(100 + k)
    nz = torch.randn((1, frames_o, model.cfg.patched_latent_dim), generator=rng).numpy()
    synth_onnx(noise=nz, timings=tm)
    if k > 0:
        for a, b in tm.items():
            acc.setdefault(a, []).append(b * 1000)
print("[speed ms, mean of 5] " + "  ".join(f"{a}={np.mean(b):.0f}" for a, b in acc.items()) + f"  | total={sum(np.mean(b) for b in acc.values()):.0f}")

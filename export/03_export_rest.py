"""DiT以外の部品(テキスト/参照声/長さ予測/コーデックのデコーダ)をONNXに書き出して、PyTorchと比較する(スパイク)。

部品ごとに独立して試し、成否と一致度を最後に表で出す。
"""

import os
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import common

common.patch_real_rope()
rt = common.load_runtime()
model = rt.model.eval()

import irodori_tts.attention as A
import irodori_tts.model as M


def _plan(segment_masks, *, query_len, attention_dtype):
    kv_mask = segment_masks[0] if len(segment_masks) == 1 else torch.cat(segment_masks, dim=1)
    return A.ContextAttentionPlan(kv_mask=kv_mask, query_len=query_len, use_fa3=False)


A.build_context_attention_plan = _plan
M.build_context_attention_plan = _plan


def _safe_attention_mask(x, mask):
    # 元実装の `if bool(has_any.all())` はデータ依存の分岐でexportできない。分岐なしの等価な書き方:
    # 有効トークンが1つも無い行だけ、xを0にして先頭をTrueにする(全行に有効トークンがあれば何も変わらない)
    mask = mask.to(device=x.device, dtype=torch.bool)
    has_any = mask.any(dim=1)
    x = torch.where(has_any[:, None, None], x, torch.zeros_like(x))
    first = torch.arange(mask.shape[1], device=mask.device)[None, :] == 0
    return x, mask | (~has_any[:, None] & first)


M._safe_attention_mask = _safe_attention_mask
# 各エンコーダのRoPE周波数は、形が動的だとキャッシュ長の分岐がガードになるので固定の長い表を使う
for mod in model.modules():
    if hasattr(mod, "_rope_freqs") and hasattr(mod, "head_dim"):
        table = M.precompute_freqs_cis(mod.head_dim, 4096)
        mod._rope_freqs = (lambda tb: (lambda seq_len, device: tb[:seq_len]))(table)

import onnxruntime as ort
from torch.export import Dim

TEXT = "おつかれさまです。ちょっと休憩しましょうか。"
ref = torch.load(os.path.join(common.REF_DIR, "dit_step_ref.pt"), weights_only=True)
results = []
ONLY = sys.argv[1:]


def want(name):
    return not ONLY or name in ONLY


def export_and_check(name, module, args, input_names, output_names, dynamic_shapes, expected, tol_note=""):
    if not want(name):
        return
    path = os.path.join(common.REF_DIR, f"{name}.onnx")
    try:
        t0 = time.perf_counter()
        with torch.no_grad():
            torch.onnx.export(module.eval(), args, path, input_names=input_names, output_names=output_names,
                              dynamic_shapes=dynamic_shapes, dynamo=True, external_data=True)
        el = time.perf_counter() - t0
        size = sum(os.path.getsize(os.path.join(common.REF_DIR, f)) for f in os.listdir(common.REF_DIR) if f.startswith(name) and not f.endswith(".pt"))
        feeds = {n: a.numpy() for n, a in zip(input_names, args)}
        row = [name, f"OK export {el:.0f}s, {size / 1e6:.0f}MB"]
        for prov in (["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]):
            sess = ort.InferenceSession(path, providers=prov)
            outs = sess.run(None, feeds)
            errs = []
            for o, e in zip(outs, expected):
                e = e.numpy()
                if e.dtype == bool:
                    errs.append(f"exact={bool((o == e).all())}")
                else:
                    errs.append(f"rel_l2={np.linalg.norm(o - e) / (np.linalg.norm(e) + 1e-12):.2e}")
            row.append(f"{prov[0][:4]}: " + ",".join(errs))
        results.append(row)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        results.append([name, f"FAILED {type(e).__name__}: {str(e)[:200]}"])


# ---- 1. テキスト: token ids -> text_state (ModernBERT-ja + projector + norm)
class TextBranch(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, mask):
        return self.m.text_norm(self.m.text_encoder(self.m.pretrained_text_backbone, input_ids, mask))


from irodori_tts.text_normalization import normalize_text

ids, mask = rt.tokenizer.batch_encode([normalize_text(TEXT).strip()])
with torch.no_grad():
    exp_text = TextBranch(model)(ids, mask)
print("text branch check vs stored reference:", float((exp_text - ref["text_state"]).abs().max()))
L = Dim("L", min=1, max=512)
export_and_check("text_encoder", TextBranch(model), (ids, mask), ["input_ids", "mask"], ["text_state"],
                 {"input_ids": {1: L}, "mask": {1: L}}, [exp_text])


# ---- 2. 参照声: patched reference latent (1,N,32)+mask -> speaker_state, speaker_mask
class SpeakerBranch(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ref_latent, ref_mask):
        seq, msk = M.patch_sequence_with_mask(seq=ref_latent, mask=ref_mask, patch_size=self.m.cfg.speaker_patch_size)
        st = self.m.speaker_norm(self.m.speaker_encoder(seq, msk))
        st, msk = self.m._prepend_masked_mean_token(st, msk)
        return st, msk


from irodori_tts.codec import patchify_latent

lat = torch.cat([torch.load(os.path.join(common.VOICES, f"{n}_lat.pt"), weights_only=True) for n in common.REF_NAMES], dim=0)
ref_latent = patchify_latent(lat.unsqueeze(0), model.cfg.latent_patch_size)
ref_mask = torch.ones(ref_latent.shape[:2], dtype=torch.bool)
with torch.no_grad():
    exp_spk, exp_spk_mask = SpeakerBranch(model)(ref_latent, ref_mask)
print("speaker branch check vs stored reference:", float((exp_spk - ref["spk_state"]).abs().max()))
N = Dim("N", min=8, max=4096)
export_and_check("speaker_encoder", SpeakerBranch(model), (ref_latent, ref_mask), ["ref_latent", "ref_mask"], ["speaker_state", "speaker_mask"],
                 {"ref_latent": {1: N}, "ref_mask": {1: N}}, [exp_spk, exp_spk_mask])


# ---- 3. 長さ予測
class DurationBranch(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, text_state, text_mask, speaker_state, speaker_mask, duration_features, has_speaker, caption_state, caption_mask, has_caption):
        return self.m.predict_duration_log_frames(
            text_state=text_state, text_mask=text_mask, speaker_state=speaker_state, speaker_mask=speaker_mask,
            duration_features=duration_features, has_speaker=has_speaker,
            caption_state=caption_state, caption_mask=caption_mask, has_caption=has_caption,
        )


from irodori_tts.duration import build_duration_features

feat = build_duration_features([normalize_text(TEXT).strip()], token_counts=ref["text_mask"].sum(dim=1), max_text_len=256, has_speaker=[True])
has_spk = torch.tensor([True])
has_cap = torch.tensor([False])
dargs = (ref["text_state"], ref["text_mask"], ref["spk_state"], ref["spk_mask"], feat, has_spk, ref["cap_state"], ref["cap_mask"], has_cap)
with torch.no_grad():
    exp_dur = DurationBranch(model)(*dargs)
print("duration log_frames:", exp_dur.tolist(), "-> frames", float(exp_dur.exp()))
S = Dim("S", min=2, max=1024)
export_and_check("duration", DurationBranch(model), dargs,
                 ["text_state", "text_mask", "speaker_state", "speaker_mask", "duration_features", "has_speaker", "caption_state", "caption_mask", "has_caption"],
                 ["log_frames"],
                 {"text_state": {1: L}, "text_mask": {1: L}, "speaker_state": {1: S}, "speaker_mask": {1: S}, "duration_features": None,
                  "has_speaker": None, "caption_state": None, "caption_mask": None, "has_caption": None}, [exp_dur])


# ---- 4. コーデックのデコーダ: latent (1,T,32) -> audio (1,1,samples)
class CodecDecode(torch.nn.Module):
    def __init__(self, codec):
        super().__init__()
        self.model = codec.model

    def forward(self, latent):
        return self.model.decode(latent.transpose(1, 2).contiguous())


with torch.no_grad():
    exp_wav = rt.codec.decode_latent(lat[:120].unsqueeze(0).float())
print("codec decode:", tuple(exp_wav.shape), "sr", rt.codec.sample_rate)
Tt = Dim("Tt", min=4, max=4096)
export_and_check("codec_decoder", CodecDecode(rt.codec), (lat[:120].unsqueeze(0).float(),), ["latent"], ["audio"], {"latent": {1: Tt}}, [exp_wav])

print("\n=== SUMMARY ===")
for r in results:
    print(" | ".join(r))

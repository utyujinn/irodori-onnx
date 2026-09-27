"""dit_step2/duration/text_encoder/caption_encoderが、初めて見る形状(文の長さ・生成フレーム数・
キャプションの長さ)で最初の1回だけ遅くならないかを確認する(TASK.md 24番 Phase 3)。

CUDAの畳み込みアルゴリズム探索をHEURISTICにしているので(engine.rsのConvAlgorithmSearch::Heuristic
と同じ設定)、毎回総当たりする既定のEXHAUSTIVEほどの極端な初回ストールは無いはずだが、それでも
「新しい形状を初めて実行したとき」対「同じ形状をもう一度実行したとき」の差を実測して確認する。

    python newshape_test.py

前提: CUDA対応GPUと、export/ref/ 以下にfp16のONNXファイル一式(text_encoder_fp16.onnx,
caption_encoder_fp16.onnx, duration_fp16.onnx, dit_step2_fp16.onnx, codec_decoder_fp16.onnx)が
必要。無ければ 03_export_rest.py / 06_fp16.py で作る。
"""

import os
import sys
import time

import numpy as np
import onnxruntime as ort

sys.stdout.reconfigure(encoding="utf-8")  # Windows console default (cp932) can't encode "—" etc.

R = os.path.join(os.path.dirname(__file__), "ref")
# engine.rsのbuild_sessionと同じ設定(HEURISTIC: 既定のEXHAUSTIVEは新しい形状ごとに総当たりベンチマーク
# してしまい、デコーダを数分単位で止める)。
PROV = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC", "arena_extend_strategy": "kSameAsRequested"}), "CPUExecutionProvider"]
RUN = ort.RunOptions()

rng = np.random.default_rng(0)
emb = lambda: rng.standard_normal((1, 512)).astype(np.float32)


def sess(name):
    return ort.InferenceSession(os.path.join(R, f"{name}_fp16.onnx"), providers=PROV)


S = {n: sess(n) for n in ("text_encoder", "caption_encoder", "duration", "dit_step2", "codec_decoder")}


def run_shape(text_len, caption_len, frames, speaker_len=149):
    """1回分のフル合成を模した実行(テキスト/キャプション/長さ予測/DiT4ステップ/デコード)。engine.rsの
    synthesize()と同じ順序・同じ入力名。"""
    ids = rng.integers(1, 1000, (1, text_len)).astype(np.int64)
    text_mask = np.ones((1, text_len), dtype=bool)
    t0 = time.perf_counter()
    text_state = S["text_encoder"].run(None, {"input_ids": ids, "mask": text_mask}, RUN)[0]

    cap_ids = rng.integers(1, 1000, (1, caption_len)).astype(np.int64)
    cap_mask = np.ones((1, caption_len), dtype=bool)
    caption_state = S["caption_encoder"].run(None, {"input_ids": cap_ids, "mask": cap_mask}, RUN)[0]

    spk = rng.standard_normal((1, speaker_len, 768)).astype(np.float32)
    spk_mask = np.ones((1, speaker_len), dtype=bool)
    S["duration"].run(
        None,
        {
            "text_state": text_state, "text_mask": text_mask, "speaker_state": spk, "speaker_mask": spk_mask,
            "duration_features": rng.standard_normal((1, 14)).astype(np.float32), "has_speaker": np.array([True]),
            "caption_state": caption_state, "caption_mask": cap_mask, "has_caption": np.array([True]),
        },
        RUN,
    )

    x = rng.standard_normal((1, frames, 32)).astype(np.float32)
    for _ in range(4):
        velocity = S["dit_step2"].run(
            None,
            {
                "x_t": x, "t_embed": emb(), "delta_embed": emb(), "text_state": text_state, "text_mask": text_mask,
                "speaker_state": spk, "speaker_mask": spk_mask, "caption_state": caption_state, "caption_mask": cap_mask,
            },
            RUN,
        )[0]
        x = x + velocity * 0.01
    S["codec_decoder"].run(None, {"latent": x}, RUN)
    return (time.perf_counter() - t0) * 1000


# ウォームアップ: CUDAコンテキスト自体の初期化コスト(形状とは無関係)をここで吸収する。
run_shape(text_len=10, caption_len=3, frames=100)

# text_len・caption_len・framesの組み合わせをすべて初めての形状にする(前後の値と全部変える)。
NEW_SHAPES = [
    (17, 5, 137), (34, 11, 264), (53, 2, 401), (8, 21, 62), (91, 7, 690),
    (25, 34, 190), (46, 4, 350), (13, 44, 99), (67, 9, 512), (22, 15, 168),
]

print(f"{'text_len':>8} {'cap_len':>7} {'frames':>7} {'1回目(ms)':>10} {'2回目(ms)':>10} {'比率':>6}")
worst_ratio = 0.0
for text_len, caption_len, frames in NEW_SHAPES:
    first = run_shape(text_len, caption_len, frames)
    second = run_shape(text_len, caption_len, frames)
    ratio = first / second if second > 0 else float("inf")
    worst_ratio = max(worst_ratio, ratio)
    print(f"{text_len:>8} {caption_len:>7} {frames:>7} {first:>10.0f} {second:>10.0f} {ratio:>5.2f}x")

print()
if worst_ratio > 1.5:
    print(f"初回が同じ形状の2回目より最大{worst_ratio:.2f}倍遅い形状があった — HEURISTICでも初回ストールが残っている可能性。cudnn_conv_algo_searchやウォームアップ方針の見直しを検討。")
else:
    print(f"初回と2回目の差は最大でも{worst_ratio:.2f}倍 — 新しい形状による目立ったストールは無いと判断してよさそう。")

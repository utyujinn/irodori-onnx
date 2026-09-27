"""テキスト前処理の正解データを作る(Rust側の一致確認用)。

各文について、Python実装の 正規化後の文字列 / トークンID(BOS付き) / 長さ予測の特徴量(14次元) を JSON に書き出す。
出力: crates/irodori-tts/tests/data/text_golden.json (モデルも参照声も含まないので、リポジトリに入れてよい)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common

rt = common.load_runtime()
from irodori_tts.duration import build_duration_features
from irodori_tts.text_normalization import normalize_text

CASES = [
    "おはよう。", "おつかれさまです。ちょっと休憩しましょうか。", "こんにちは、私はAIです。これは音声合成のテストです",
    "「こんにちは」", "『二重』の「括弧」", "（かっこ）", "「a」「b」", "「入れ子「の」中」", "((x))",
    "えー…本当に……？信じられない…………！", "うそ...ほんと..?", "ｱｲｳｴｵ ｶﾞｷﾞｸﾞ", "ＡＢＣ１２３ａｂｃ", "Hello, World! 123", "tab\tと　全角スペース",
    "～波ダッシュ〜と全角チルダ", "ー長音ーーー", "ー", "！？", "？！", "♥ハート●丸◯白〇", "①②③④⑤⑥", "▼記号《二重》≪山≫♀♂;セミコロン",
    "ダッシュ‐‑‒–—―と−マイナス", "[n]改行[n]記号", "\\[n\\]バックスラッシュ",
    "ありがとう😊", "😊前置き", "えっ😲まじで😮‍💨ふう🥺", "ひそひそ👂こそこそ", "笑い🤭くすくす😆あはは", "😮‍💨😮‍💨", "⏩早口で⏸️間を置いて🐢ゆっくり",
    "漢字とかなとカナと123とabc", "東京都渋谷区神宮前", "𠮷野家", "abc", "1", "あ", "。", "、、、、、", "!!!!!!!!!!", "??????????",
    "長文テスト。" * 20,
    "おはよう。徳川家の将軍様に使えるのはとてもいいことです。毎日報酬として経験値をいただくことができます。",
    "カタカナだけのテキストデス", "ﾊﾝｶｸｶﾅ", "ｗｗｗ草", "１２３４５", "  前後に空白  ", "改行\nを含む\nテキスト",
]

MAX_TEXT_LEN = rt.default_text_max_len
out = {"max_text_len": MAX_TEXT_LEN, "cases": []}
for raw in CASES:
    norm = normalize_text(raw).strip()
    if norm == "":
        out["cases"].append({"raw": raw, "normalized": norm, "empty": True})
        continue
    ids = rt.tokenizer.encode(norm).tolist()  # BOS付き、add_special_tokens=False
    feat = build_duration_features([norm], token_counts=[len(ids)], max_text_len=MAX_TEXT_LEN, has_speaker=[True])[0].tolist()
    out["cases"].append({"raw": raw, "normalized": norm, "token_ids": ids, "duration_features": feat})

dst = os.path.join(os.path.dirname(__file__), "..", "crates", "irodori-tts", "tests", "data", "text_golden.json")
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"{len(out['cases'])} cases -> {os.path.normpath(dst)}  (bos_token_id={rt.tokenizer.bos_token_id}, max_text_len={MAX_TEXT_LEN})")

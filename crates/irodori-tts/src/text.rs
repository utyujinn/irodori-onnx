//! Text preprocessing that the Python runtime does before tokenization: `normalize_text`
//! (irodori_tts/text_normalization.py) and the 14 hand-made duration features (irodori_tts/duration.py).

use unicode_normalization::UnicodeNormalization;

/// Bracket pairs that are stripped when they enclose the whole text.
const BRACKET_PAIRS: [(char, char); 5] = [('「', '」'), ('『', '』'), ('（', '）'), ('【', '】'), ('(', ')')];

/// Same steps, in the same order, as Python's `normalize_text`.
pub fn normalize_text(text: &str) -> String {
    // The replacement order matters: "[n]" is removed before the escaped variant.
    let mut s = text.to_string();
    for (old, new) in [
        ("\t", ""),
        ("[n]", ""),
        ("\\[n\\]", ""),
        ("\u{3000}", ""),
        ("？", "?"),
        ("！", "!"),
        ("♥", "♡"),
        ("●", "○"),
        ("◯", "○"),
        ("〇", "○"),
    ] {
        s = s.replace(old, new);
    }

    let mut out = String::with_capacity(s.len());
    let mut ellipsis_run = 0usize;
    for c in s.chars() {
        // `…{3,}` -> `……`: count the run and flush it when it ends.
        if c == '…' {
            ellipsis_run += 1;
            continue;
        }
        flush_ellipsis(&mut out, &mut ellipsis_run);
        match c {
            ';' | '▼' | '♀' | '♂' | '《' | '》' | '≪' | '≫' | '①'..='⑥' => {}
            '\u{02d7}' | '\u{2010}'..='\u{2015}' | '\u{2043}' | '\u{2212}' | '\u{23af}' | '\u{23e4}' | '\u{2500}' | '\u{2501}'
            | '\u{2e3a}' | '\u{2e3b}' => {}
            '\u{ff5e}' | '\u{301c}' => out.push('ー'),
            other => out.push(other),
        }
    }
    flush_ellipsis(&mut out, &mut ellipsis_run);

    let s = strip_outer_brackets(&out);
    let s: String = s.nfkc().collect();
    s.replace("...", "…").replace("..", "…")
}

fn flush_ellipsis(out: &mut String, run: &mut usize) {
    // Runs of 1 or 2 are left as they are, runs of 3 or more become exactly two.
    let n = if *run >= 3 { 2 } else { *run };
    for _ in 0..n {
        out.push('…');
    }
    *run = 0;
}

fn strip_outer_brackets(text: &str) -> String {
    let mut chars: Vec<char> = text.chars().collect();
    loop {
        if chars.len() < 2 {
            break;
        }
        let (start, end) = (chars[0], chars[chars.len() - 1]);
        let Some(&(_, close)) = BRACKET_PAIRS.iter().find(|(open, _)| *open == start) else {
            break;
        };
        if close != end {
            break;
        }
        let mut depth = 0i32;
        let mut encloses_all = true;
        for (i, &c) in chars.iter().enumerate() {
            if c == start {
                depth += 1;
            } else if c == end {
                depth -= 1;
            }
            if depth == 0 && i < chars.len() - 1 {
                encloses_all = false;
                break;
            }
        }
        if encloses_all && depth == 0 {
            chars = chars[1..chars.len() - 1].to_vec();
            continue;
        }
        break;
    }
    chars.into_iter().collect()
}

/// Emojis the model treats as style annotations. The order is the one in the Python source: the matcher tries the longest
/// first and, among equally long ones, keeps this order (Python sorts them by length with a stable sort).
const ANNOTATION_EMOJIS: [&str; 56] = [
    "⏩", "⏱\u{fe0f}", "⏸\u{fe0f}", "🌬\u{fe0f}", "🍭", "🎛\u{fe0f}", "🎭", "🎵", "🐢", "🐱", "👂", "👃", "👅", "👌", "👏", "💋", "💥", "💦",
    "💪", "📄", "📞", "📢", "📣", "😆", "😊", "😌", "😎", "😏", "😒", "😖", "😟", "😠", "😪", "😭", "😮", "😮\u{200d}💨", "😰", "😱", "😲",
    "😴", "🙄", "🙏", "🤐", "🤔", "🤢", "🤧", "🤭", "🥤", "🥱", "🥴", "🥵", "🥹", "🥺", "🫣", "🫶", "📖",
];

fn count_annotation_emojis(text: &str) -> usize {
    let mut sorted: Vec<&str> = ANNOTATION_EMOJIS.to_vec();
    sorted.sort_by_key(|e| std::cmp::Reverse(e.chars().count()));
    let mut count = 0;
    let mut i = 0;
    while i < text.len() {
        if let Some(e) = sorted.iter().find(|e| text[i..].starts_with(**e)) {
            count += 1;
            i += e.len();
        } else {
            i += text[i..].chars().next().map_or(1, char::len_utf8);
        }
    }
    count
}

fn log1p_cap(count: usize, cap: usize) -> f64 {
    (count.min(cap) as f64).ln_1p() / (cap as f64).ln_1p()
}

fn is_kana(c: char) -> bool {
    matches!(c as u32, 0x3040..=0x309f | 0x30a0..=0x30ff)
}

fn is_kanji(c: char) -> bool {
    matches!(c as u32, 0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xf900..=0xfaff | 0x20000..=0x2fa1f)
}

/// The 14 features fed to the duration predictor, in the order of Python's `build_duration_features`.
/// `token_count` counts the BOS token as well (it is the number of valid tokens in the text mask).
pub fn duration_features(text: &str, token_count: usize, max_text_len: usize, has_speaker: bool) -> [f32; 14] {
    let char_count = text.chars().count().max(1) as f64;
    let count = |chars: &[char]| text.chars().filter(|c| chars.contains(c)).count();
    let kana = text.chars().filter(|&c| is_kana(c)).count() as f64;
    let kanji = text.chars().filter(|&c| is_kanji(c)).count() as f64;
    let alnum = text.chars().filter(|c| c.is_ascii_alphanumeric()).count() as f64;
    let max_len = max_text_len as f64;
    let features = [
        (token_count as f64).clamp(0.0, max_len) / max_len,
        char_count.min(512.0).ln_1p() / 512f64.ln_1p(),
        token_count as f64 / char_count,
        log1p_cap(count(&['。', '.']), 8),
        log1p_cap(count(&['、', ',']), 16),
        log1p_cap(count(&['ー']), 8),
        log1p_cap(count(&['…']), 8),
        log1p_cap(count(&['！', '!']), 8),
        log1p_cap(count(&['？', '?']), 8),
        log1p_cap(count_annotation_emojis(text), 8),
        kana / char_count,
        kanji / char_count,
        alnum / char_count,
        if has_speaker { 1.0 } else { 0.0 },
    ];
    features.map(|f| f as f32)
}

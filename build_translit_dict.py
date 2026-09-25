"""
Build a Latin <-> native-script token dictionary FROM THE TRAINING POSITIVES ONLY.

Why: the real examples show transliterations are faithful and word-for-word
('Global Investment Private Limited' <-> 'ग्लोबल इन्वेस्टमेंट प्रा. लि.').
When a Latin name and its native-script match have the SAME token count, we can
align them positionally and accumulate (native_token -> latin_token) counts across
the whole training set. No external data, no lookups — every pair used already
carries a ground-truth label.

This covers all 9 Indic scripts uniformly (Devanagari, Tamil, Kannada, Telugu,
Malayalam, Bengali, Gujarati, Odia, Gurmukhi) with ONE mechanism, instead of
hand-writing nine legal-suffix dictionaries. French gets no dictionary here —
there are zero training pairs to mine for it — so French relies on the
hand-written accent-folding + suffix list instead.

Usage:
    python build_translit_dict.py --data-dir dataset --out translit_dict.json
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd

_NATIVE_RE = re.compile(r"[\u0900-\u0D7F]")   # Devanagari .. Malayalam (matches the census)
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # fine for pure-Latin text only
# IMPORTANT: Python's `\w` does NOT include Indic combining vowel signs (Unicode
# category Mn), so a \w-based regex shreds Devanagari/Tamil/etc. words at every
# matra (e.g. 'टेक्नोलॉजीज' -> 16 fragments instead of 1 token). Native-script
# words are split on whitespace instead, then stripped of surrounding punctuation
# only -- this matches how these scripts actually separate words in the real data.
_PUNCT_STRIP = re.compile(r"^[^\w\u0900-\u0D7F]+|[^\w\u0900-\u0D7F]+$", re.UNICODE)


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False, quoting=3)


def is_native(text: str) -> bool:
    return bool(_NATIVE_RE.search(text))


def tokens(text: str):
    """Whitespace-split, punctuation-stripped, drop pure-digit/empty tokens.
    Works uniformly for Latin ('Global') and Indic ('टेक्नोलॉजीज') text."""
    out = []
    for raw in text.casefold().split():
        t = _PUNCT_STRIP.sub("", raw)
        if t and not t.isdigit():
            out.append(t)
    return out


def build(data_dir: str, min_count: int = 3):
    tr = os.path.join(data_dir, "train")
    gt = read_tsv(os.path.join(tr, "train_ground_truth.tsv"))
    gt["ids"] = gt["matched_entity_ids"].map(lambda s: [x.strip() for x in s.split(",") if x.strip()])

    names = {}
    for i in (1, 2, 3):
        df = read_tsv(os.path.join(tr, f"train_source{i}.tsv"))
        names.update(dict(zip(df["entity_id"], df["business_name"])))

    counts = defaultdict(Counter)   # native_token -> Counter(latin_token -> n)
    aligned, skipped_len_mismatch, no_native = 0, 0, 0

    for _, row in gt.iterrows():
        s1_name = names.get(row["source1_entity_id"], "")
        if is_native(s1_name):
            continue  # S1 is the clean reference source; always expected to be Latin
        latin_toks = tokens(s1_name)
        if not latin_toks:
            continue
        for mid in row["ids"]:
            m_name = names.get(mid)
            if m_name is None or not is_native(m_name):
                no_native += 1
                continue
            native_toks = tokens(m_name)
            if len(native_toks) != len(latin_toks):
                skipped_len_mismatch += 1
                continue
            for lt, nt in zip(latin_toks, native_toks):
                counts[nt][lt] += 1
            aligned += 1

    # keep only the top Latin token per native token, and only if seen often enough
    dictionary = {
        nt: c.most_common(1)[0][0]
        for nt, c in counts.items()
        if sum(c.values()) >= min_count
    }
    stats = {
        "aligned_name_pairs": aligned,
        "skipped_token_count_mismatch": skipped_len_mismatch,
        "matches_with_no_native_script": no_native,
        "distinct_native_tokens_kept": len(dictionary),
        "distinct_native_tokens_seen": len(counts),
    }
    return dictionary, stats


def apply_dict(text: str, dictionary: dict) -> str:
    """Best-effort: replace each native token with its learned Latin token, else keep as-is."""
    pieces = text.split(" ")
    return " ".join(dictionary.get(_PUNCT_STRIP.sub("", p.casefold()), p) for p in pieces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out", default="translit_dict.json")
    ap.add_argument("--min-count", type=int, default=3,
                     help="drop a native token's mapping unless seen at least this often")
    a = ap.parse_args()
    d, stats = build(a.data_dir, a.min_count)
    print(json.dumps(stats, indent=2))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    print(f"\nwrote {len(d)} native->latin token mappings to {a.out}")
    sample = list(d.items())[:15]
    print("sample:", sample)


if __name__ == "__main__":
    main()

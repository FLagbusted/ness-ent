#!/usr/bin/env python3
"""
Script / language / noise census for the Business Entity Resolution challenge.

Answers, before any modelling:
  1. Which countries, Unicode scripts and (hinted) languages occur in which file?
  2. What appears in TEST that never appears in TRAIN (unseen scripts/characters/countries)?
  3. How much of each noise pattern is there (NULL tokens, ALL CAPS, domain-like names, ...)?
  4. Ground-truth structure: singletons, matches per S1, does an S2/S3 record ever have
     two owners, do true pairs cross country labels, how many true pairs are cross-script?

Usage (from the folder that contains dataset/):
    python er_script_census.py --data-dir dataset --out eda_out
    python er_script_census.py --data-dir dataset --out eda_out --lingua   # optional extra

Needs only pandas. --lingua additionally needs: pip install lingua-language-detector
Nothing here calls any external service.
"""
import argparse
import os
import re
import unicodedata
from collections import Counter

import pandas as pd

# ----------------------------------------------------------------------------- scripts
KNOWN = {
    "LATIN", "DEVANAGARI", "TAMIL", "KANNADA", "TELUGU", "MALAYALAM", "BENGALI", "GUJARATI",
    "GURMUKHI", "ORIYA", "SINHALA", "ARABIC", "CYRILLIC", "GREEK", "HEBREW", "THAI", "HANGUL",
    "HIRAGANA", "KATAKANA", "CJK", "ARMENIAN", "GEORGIAN", "TIBETAN", "MYANMAR", "KHMER", "LAO",
    "ETHIOPIC",
}
SCRIPT_LANG_HINT = {
    "Latin": "English / French / romanised text (script alone cannot tell)",
    "Devanagari": "Hindi / Marathi / Nepali (shared script)",
    "Tamil": "Tamil", "Kannada": "Kannada", "Telugu": "Telugu", "Malayalam": "Malayalam",
    "Bengali": "Bengali / Assamese", "Gujarati": "Gujarati", "Gurmukhi": "Punjabi",
    "Odia": "Odia", "Arabic": "Arabic / Urdu / Persian", "Cyrillic": "Russian / other Cyrillic",
    "Greek": "Greek", "CJK": "Chinese / Japanese kanji", "Hangul": "Korean",
    "Hiragana": "Japanese", "Katakana": "Japanese", "Thai": "Thai", "Hebrew": "Hebrew",
}
_UNSET = object()
_CACHE = {}


def char_script(ch):
    """Script of one character; None for digits/punctuation/space/combining accents."""
    hit = _CACHE.get(ch, _UNSET)
    if hit is not _UNSET:
        return hit
    res = None
    if ch.isalpha() or unicodedata.category(ch) in ("Mn", "Mc"):
        if ord(ch) < 128:
            res = "Latin"
        else:
            try:
                first = unicodedata.name(ch).split(" ")[0]
            except ValueError:
                first = ""
            if first in KNOWN:
                res = {"ORIYA": "Odia", "CJK": "CJK"}.get(first, first.capitalize())
            elif first != "COMBINING":
                res = "Other"
    _CACHE[ch] = res
    return res


def scripts_of(text):
    return {s for s in map(char_script, text) if s}


def profile(text):
    sc = scripts_of(text)
    return "+".join(sorted(sc)) if sc else "(none)"


def coarse(p):
    return "Latin" if p == "Latin" else ("(none)" if p == "(none)" else "non-Latin/mixed")


def has_latin_accent(text):
    return any(ord(c) > 127 and char_script(c) == "Latin" for c in text)


def mixed_inside_token(text):
    return any(len(scripts_of(t)) > 1 for t in text.split())


# ------------------------------------------------------------------- noise / locale flags
RE_NULL = r"\bnull\b|\bnan\b|\bn/?a\b"
RE_DOMAIN = r"\.(?:com|net|org|in|co|io|biz|info|fr)\b"
RE_BRACKET = r"[\[\]]"
RE_LETTER_DIGIT = r"[A-Za-z]{2,}\d+[A-Za-z]*|\d+[A-Za-z]{2,}"
RE_STATE_FIRST = r"^\s*[A-Za-z]{2},\s"
FR_WORDS = {"rue", "sarl", "sas", "sasu", "eurl", "snc", "sci", "chemin", "impasse", "allee",
            "groupe", "comite", "federation", "societe", "pharmacie", "ecole", "etablissements", "ets"}
RE_FR_STREET = re.compile(r"^\W*\d+\W*\s+(?:rue|r|av|bd|chemin|impasse|all[ée]e|quai)\b", re.I)


def allcaps(t):
    letters = [c for c in t if c.isascii() and c.isalpha()]
    return len(letters) >= 4 and all(c.isupper() for c in letters)


def fr_like(name, addr):
    toks = set(re.findall(r"[^\W\d_]+", (name + " " + addr).lower()))
    plain = {unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode() for t in toks}
    return bool(plain & FR_WORDS) or bool(RE_FR_STREET.search(addr)) or has_latin_accent(name + addr)


def enrich(df):
    n, a = df["business_name"], df["business_address"]
    df["country"] = df["country"].str.strip().replace("", "(missing)")
    df["name_script"] = n.map(profile)
    df["addr_script"] = a.map(profile)
    df["name_multi_script"] = n.map(lambda t: len(scripts_of(t)) > 1)
    df["name_mixed_in_token"] = n.map(mixed_inside_token)
    df["latin_accent"] = (n + " " + a).map(has_latin_accent)
    df["fr_like_heuristic"] = [fr_like(x, y) for x, y in zip(n, a)]
    df["f_name_empty"] = n.str.strip().eq("")
    df["f_addr_empty"] = a.str.strip().eq("")
    df["f_addr_null_token"] = a.str.contains(RE_NULL, case=False, regex=True)
    df["f_addr_hash"] = a.str.contains("#", regex=False)
    df["f_addr_state_first"] = a.str.match(RE_STATE_FIRST)
    df["f_addr_allcaps"] = a.map(allcaps)
    df["f_name_allcaps"] = n.map(allcaps)
    df["f_name_domain_like"] = n.str.contains(RE_DOMAIN, case=False, regex=True)
    df["f_name_bracket"] = n.str.contains(RE_BRACKET, regex=True)
    df["f_name_letter_digit"] = n.str.contains(RE_LETTER_DIGIT, regex=True)
    return df


# --------------------------------------------------------------------------------- I/O
def read_tsv(path):
    # QUOTE_NONE (3): stray quote characters in names must not swallow rows.
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                       quoting=3, encoding="utf-8", on_bad_lines="warn")


def load_all(data_dir):
    frames = {}
    for split in ("train", "test"):
        for src in (1, 2, 3):
            p = os.path.join(data_dir, split, f"{split}_source{src}.tsv")
            if not os.path.exists(p):
                print(f"[skip] missing {p}")
                continue
            df = read_tsv(p)
            for c in ("business_name", "business_address", "country"):
                if c not in df.columns:
                    print(f"[warn] {p} has no '{c}' column; treating as empty")
                    df[c] = ""
            print(f"[{split} S{src}] rows={len(df)} raw country labels={sorted(map(repr, df['country'].unique()))}")
            df["split"], df["src"] = split, f"S{src}"
            frames[(split, src)] = enrich(df)
    return frames


def show(title, obj, out, fname=None):
    print(f"\n=== {title} ===")
    print(obj.to_string() if hasattr(obj, "to_string") else obj)
    if fname and hasattr(obj, "to_csv"):
        obj.to_csv(os.path.join(out, fname))


# ------------------------------------------------------------------------------ reports
def census(frames, out):
    alldf = pd.concat(frames.values(), ignore_index=True)
    keys = ["split", "src", "country"]

    show("Rows per split / source / country",
         alldf.groupby(keys).size().unstack("country", fill_value=0), out, "rows_by_country.csv")

    for col, tag in (("name_script", "NAME"), ("addr_script", "ADDRESS")):
        tab = pd.crosstab([alldf[k] for k in keys], alldf[col], normalize="index").mul(100).round(1)
        show(f"{tag} script profile, % of rows", tab, out, f"{col}_pct.csv")

    show("Every script seen (chars, name+address) and the language family it hints at",
         pd.DataFrame(sorted(
             [(s, c, SCRIPT_LANG_HINT.get(s, "?"))
              for s, c in Counter(ch_s for t in (alldf["business_name"] + " " + alldf["business_address"])
                                  for ch_s in map(char_script, t) if ch_s).items()],
             key=lambda x: -x[1]), columns=["script", "n_chars", "language_hint"]),
         out, "scripts_seen.csv")

    cols = ["name_multi_script", "name_mixed_in_token", "latin_accent", "fr_like_heuristic"]
    show("Mixed-script / accent / French-like heuristics, % of rows",
         alldf.groupby(keys)[cols].mean().mul(100).round(1), out, "script_mix_pct.csv")

    flag_cols = [c for c in alldf.columns if c.startswith("f_")]
    show("Noise flags, % of rows", alldf.groupby(keys)[flag_cols].mean().mul(100).round(1),
         out, "noise_flags_pct.csv")

    # ---- train vs test shift
    tr, te = alldf[alldf.split == "train"], alldf[alldf.split == "test"]
    if len(tr) and len(te):
        print("\n=== Train vs test: what is NEW in test ===")
        print("countries only in test :", sorted(set(te.country) - set(tr.country)))
        s_tr = set().union(*tr["name_script"].map(lambda p: set(p.split("+"))))
        s_te = set().union(*te["name_script"].map(lambda p: set(p.split("+"))))
        a_tr = set().union(*tr["addr_script"].map(lambda p: set(p.split("+"))))
        a_te = set().union(*te["addr_script"].map(lambda p: set(p.split("+"))))
        print("name scripts only in test   :", sorted(s_te - s_tr - {"(none)"}))
        print("address scripts only in test:", sorted(a_te - a_tr - {"(none)"}))

        def nonascii(df):
            return Counter(c for t in (df["business_name"] + " " + df["business_address"])
                           for c in t if ord(c) > 127)
        c_tr, c_te = nonascii(tr), nonascii(te)
        new = [(c, n, unicodedata.name(c, "?")) for c, n in c_te.most_common() if c not in c_tr]
        show("Non-ASCII characters in test never seen in train (top 40)",
             pd.DataFrame(new[:40], columns=["char", "count_in_test", "unicode_name"]),
             out, "test_only_chars.csv")


def lingua_report(frames, out):
    try:
        from lingua import LanguageDetectorBuilder
    except ImportError:
        print("\n[lingua] not installed -> pip install lingua-language-detector (optional)")
        return
    det = LanguageDetectorBuilder.from_all_languages().with_minimum_relative_distance(0.2).build()
    rows = []
    for (split, src), df in frames.items():
        latin = df[(df.name_script == "Latin")]
        for country, g in latin.groupby("country"):
            langs = Counter()
            for text in (g["business_name"] + ", " + g["business_address"]).head(3000):
                lang = det.detect_language_of(text)
                langs[lang.name if lang else "UNSURE"] += 1
            for lang, n in langs.most_common(6):
                rows.append((split, f"S{src}", country, lang, n))
    show("Lingua guess on Latin-script rows (short strings: treat as a hint only)",
         pd.DataFrame(rows, columns=["split", "src", "country", "language", "n"]),
         out, "lingua_guess.csv")


def gt_report(frames, data_dir, out):
    p = os.path.join(data_dir, "train", "train_ground_truth.tsv")
    if not os.path.exists(p) or not all(("train", s) in frames for s in (1, 2, 3)):
        print("\n[skip] ground-truth report (need train sources + train_ground_truth.tsv)")
        return
    gt = read_tsv(p)
    gt["ids"] = gt["matched_entity_ids"].map(lambda s: [x.strip() for x in s.split(",") if x.strip()])
    n = gt["ids"].map(len)
    print("\n=== Ground truth structure ===")
    print(f"S1 entities: {len(gt)} | singletons (no matches): {(n == 0).mean() * 100:.1f}%")
    print("matches per S1 (count of S1 entities):")
    print(n.value_counts().sort_index().head(12).to_string())
    print(f"S1 with S2 matches: {gt['ids'].map(lambda l: any(x.startswith('S2-') for x in l)).mean() * 100:.1f}% | "
          f"with S3 matches: {gt['ids'].map(lambda l: any(x.startswith('S3-') for x in l)).mean() * 100:.1f}%")

    pairs = gt[["source1_entity_id", "ids"]].explode("ids").dropna().rename(columns={"ids": "match_id"})
    owners = pairs["match_id"].value_counts()
    print(f"S2/S3 ids owned by MORE than one S1: {(owners > 1).sum()} of {len(owners)} matched ids "
          f"(0 means the one-owner rule holds)")
    for s in (2, 3):
        ids = set(frames[("train", s)]["entity_id"])
        print(f"S{s} train records that match no S1: {len(ids - set(owners.index)) / max(len(ids), 1) * 100:.1f}%")

    info = pd.concat([frames[("train", s)][["entity_id", "country", "name_script", "addr_script"]]
                      for s in (1, 2, 3)]).set_index("entity_id")
    pairs = pairs[pairs["match_id"].isin(info.index) & pairs["source1_entity_id"].isin(info.index)]
    a, b = info.loc[pairs["source1_entity_id"]].reset_index(drop=True), info.loc[pairs["match_id"]].reset_index(drop=True)
    print(f"true pairs whose country labels differ: {(a.country != b.country).mean() * 100:.2f}%")
    pairs = pairs.reset_index(drop=True)
    pairs["other_src"] = pairs["match_id"].str[:2]
    for col, tag in (("name_script", "NAME"), ("addr_script", "ADDRESS")):
        t = pd.DataFrame({"src": pairs["other_src"], "S1": a[col].map(coarse), "matched": b[col].map(coarse)})
        show(f"True pairs, {tag} script: S1 vs matched record (rows = S2/S3)",
             pd.crosstab(t["src"], [t["S1"], t["matched"]]), out, f"gt_pair_{col}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out", default="eda_out")
    ap.add_argument("--lingua", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    frames = load_all(args.data_dir)
    if not frames:
        raise SystemExit("No files found. Check --data-dir (expects train/ and test/ inside).")
    census(frames, args.out)
    gt_report(frames, args.data_dir, args.out)
    if args.lingua:
        lingua_report(frames, args.out)
    print(f"\nCSV copies written to {args.out}/")


if __name__ == "__main__":
    main()

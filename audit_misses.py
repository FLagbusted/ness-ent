"""
Blocking miss audit -- WHY does blocking miss the true matches it misses?

Recall has plateaued around 92% even with a TF-IDF pass and topk=250 (equal
to v2's UNPRUNED union), so the misses happen BEFORE pruning: the existing
keys never retrieve them at all. Before building another retrieval pass,
this measures what the missed pairs actually look like, so the next pass
targets a measured failure mode instead of a guessed one.

Key hypothesis tested: most misses sit in S1 entities where OTHER true
matches WERE found ("partial recall": easy siblings make the row look
strong, so no per-row 'is this row weak?' criterion can flag it). If a
missed record is similar to one of its FOUND siblings, then using found
siblings as extra queries ("query expansion") would recover it.

Per missed (S1, candidate) pair it records:
  - n_true / n_found for that S1 (partial vs total miss)
  - name similarity S1 <-> missed (Jaro-Winkler, token jaccard, after the
    same latinization + cleaning train_matcher.py uses)
  - address token jaccard + numeric-suffix overlap S1 <-> missed
  - native-script mismatch flag
  - best name/address similarity missed <-> ANY FOUND SIBLING  (the
    query-expansion test)
and assigns a heuristic bucket. Writes miss_audit.tsv (all misses) and
prints a summary.

Usage:
    python audit_misses.py --candidates cand_train.parquet --data-dir dataset \
        --split train --translit-dict translit_dict.json --out miss_audit.tsv
"""
import argparse
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler

from normalize import numeric_suffix_overlap
from train_matcher import filtered_read_tsv, build_entity_index, jaccard, read_tsv


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def name_sims(a: dict, b: dict):
    """best-of-variants JW + token jaccard, same convention as train_matcher."""
    jw = jac = 0.0
    for va in a["variants"]:
        for vb in b["variants"]:
            jw = max(jw, JaroWinkler.normalized_similarity(va["casefolded"], vb["casefolded"]))
            jac = max(jac, jaccard(va["tokens"], vb["tokens"]))
    return jw, jac


def addr_sims(a: dict, b: dict):
    return (jaccard(a["addr_tokens"], b["addr_tokens"]),
            numeric_suffix_overlap(a["addr_numeric"], b["addr_numeric"]))


def bucket(r) -> str:
    if r["either_empty_name"]:
        return "empty_name"
    if r["script_mismatch"]:
        return "native_script_mismatch"
    if r["name_jw"] >= 0.90 or r["name_jac"] >= 0.5:
        return "name_similar_but_missed"      # retrieval/cap/bucket-drop issue
    if r["addr_jac"] >= 0.5 or r["addr_num_overlap"] > 0:
        return "address_only_link"            # name diverged (alias/rename)
    return "hard_other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="cand_train.parquet")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--translit-dict", default="translit_dict.json")
    ap.add_argument("--sibling-jw", type=float, default=0.90,
                    help="missed record counts as 'sibling-recoverable' if its name JW to any found "
                         "sibling is >= this, OR its address jaccard to any found sibling is >= 0.6")
    ap.add_argument("--out", default="miss_audit.tsv")
    a = ap.parse_args()

    with open(a.translit_dict, encoding="utf-8") as f:
        translit_dict = json.load(f)
    d = os.path.join(a.data_dir, a.split)

    log("loading candidates + ground truth ...")
    cand = pd.read_parquet(a.candidates)
    has_flag = "tfidf_ran" in cand.columns
    cand_sets = {s1: set(ids) for s1, ids in zip(cand["source1_entity_id"], cand["candidate_entity_ids"])}
    tfidf_ran = dict(zip(cand["source1_entity_id"], cand["tfidf_ran"])) if has_flag else {}
    del cand

    gt = read_tsv(os.path.join(d, f"{a.split}_ground_truth.tsv"))
    true_map = {}
    for s1, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if s1 in cand_sets:
            ts = {x.strip() for x in ids.split(",") if x.strip()}
            if ts:
                true_map[s1] = ts
    log(f"S1 in candidates with >=1 true match: {len(true_map):,} "
        f"(singletons excluded -- should match the blocking log's n)")

    missed_pairs, found_by_s1 = [], {}
    for s1, ts in true_map.items():
        found = ts & cand_sets[s1]
        found_by_s1[s1] = found
        for m in ts - found:
            missed_pairs.append((s1, m))
    n_true_total = sum(len(v) for v in true_map.values())
    log(f"missed pairs: {len(missed_pairs):,} of {n_true_total:,} true pairs "
        f"({len(missed_pairs) / max(1, n_true_total) * 100:.2f}%) across "
        f"{len({s for s, _ in missed_pairs}):,} S1 entities")
    if not missed_pairs:
        return

    need_s1 = {s for s, _ in missed_pairs}
    need_other = {m for _, m in missed_pairs}
    for s in need_s1:
        need_other |= found_by_s1[s]
    log(f"reading {len(need_s1):,} S1 + {len(need_other):,} S2/S3 records (stream-filtered) ...")
    s1_idx = build_entity_index(filtered_read_tsv(os.path.join(d, f"{a.split}_source1.tsv"), need_s1), translit_dict)
    oth = build_entity_index(filtered_read_tsv(os.path.join(d, f"{a.split}_source2.tsv"), need_other), translit_dict)
    oth.update(build_entity_index(filtered_read_tsv(os.path.join(d, f"{a.split}_source3.tsv"), need_other), translit_dict))
    raw_names = {}
    for src in ("source1", "source2", "source3"):
        df = filtered_read_tsv(os.path.join(d, f"{a.split}_{src}.tsv"), need_s1 | need_other)
        for eid, n, ad in zip(df["entity_id"], df["business_name"], df["business_address"]):
            raw_names[eid] = (n, ad)
        del df

    log("scoring missed pairs ...")
    rows = []
    for s1, m in missed_pairs:
        if s1 not in s1_idx or m not in oth:
            continue
        A, B = s1_idx[s1], oth[m]
        jw, jac = name_sims(A, B)
        ajac, anum = addr_sims(A, B)
        sib_name = sib_addr = 0.0
        for f in found_by_s1[s1]:
            if f in oth:
                sjw, _ = name_sims(oth[f], B)
                sa, _ = addr_sims(oth[f], B)
                sib_name, sib_addr = max(sib_name, sjw), max(sib_addr, sa)
        r = {
            "source1_entity_id": s1, "missed_entity_id": m,
            "missed_source": m.split("-")[0] if "-" in m else "?",
            "n_true": len(true_map[s1]), "n_found": len(found_by_s1[s1]),
            "tfidf_ran": tfidf_ran.get(s1, None),
            "name_jw": round(jw, 3), "name_jac": round(jac, 3),
            "addr_jac": round(ajac, 3), "addr_num_overlap": anum,
            "script_mismatch": int(A["is_native_name"] != B["is_native_name"]),
            "either_empty_name": int(not any(v["squashed"] for v in A["variants"])
                                     or not any(v["squashed"] for v in B["variants"])),
            "sib_best_name_jw": round(sib_name, 3), "sib_best_addr_jac": round(sib_addr, 3),
            "s1_name": raw_names.get(s1, ("", ""))[0], "s1_addr": raw_names.get(s1, ("", ""))[1],
            "missed_name": raw_names.get(m, ("", ""))[0], "missed_addr": raw_names.get(m, ("", ""))[1],
        }
        r["bucket"] = bucket(r)
        r["sibling_recoverable"] = int(r["n_found"] > 0 and (sib_name >= a.sibling_jw or sib_addr >= 0.6))
        rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv(a.out, sep="\t", index=False)
    n = len(out)
    log(f"wrote {n:,} scored misses to {a.out}")

    print("\n=== WHERE the misses sit ===")
    partial = (out["n_found"] > 0).mean() * 100
    print(f"  in S1s where >=1 OTHER true match WAS found (partial recall): {partial:.1f}%")
    print(f"  in S1s where NOTHING was found (total miss):                 {100 - partial:.1f}%")
    if has_flag:
        tr = out["tfidf_ran"].fillna(False).astype(bool).mean() * 100
        print(f"  TF-IDF actually ran for the S1 (read from blocking's own flag): {tr:.1f}%")
    else:
        print("  (candidates parquet has no tfidf_ran column -- rerun blocking_recall_v5.py to get it)")

    print("\n=== WHAT the misses look like (heuristic buckets) ===")
    for b, c in Counter(out["bucket"]).most_common():
        print(f"  {b:28s} {c:7,}  ({c / n * 100:5.1f}%)")

    print("\n=== QUERY-EXPANSION TEST ===")
    sr = out["sibling_recoverable"].mean() * 100
    print(f"  misses similar to a FOUND sibling (name JW>={a.sibling_jw} or addr jac>=0.6): {sr:.1f}%")
    print("  -> this is roughly the share a sibling-query-expansion pass could plausibly recover")

    print("\n=== by source ===")
    for s, c in Counter(out["missed_source"]).most_common():
        print(f"  {s}: {c:,} ({c / n * 100:.1f}%)")
    print("\nEyeball 30 random rows:  shuf -n 30 miss_audit.tsv | cut -f5,7,9,13,15-20")


if __name__ == "__main__":
    main()

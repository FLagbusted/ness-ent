"""
Feature engineering + LightGBM baseline for the Business Entity Resolution
challenge, built on top of the pruned candidates from blocking_recall_v4.py.

Deliberately self-contained: features are recomputed fresh from RAW
business_name/business_address text (not from blocking's internal
normalized columns), so this script's correctness doesn't depend on
blocking's internals and the features stay meaningful even if blocking
changes. `candidate_agreement` (how many distinct key families blocking
matched on) is the one number reused directly from the candidates parquet
-- it's a real corroboration signal, not blocking plumbing.

Pipeline:
  1. explode candidates.parquet into (S1, candidate) pairs
  2. label = 1 if the pair is a true match (train split only)
  3. entity-level train/val split (never split a single S1's pairs across
     both -- a pairwise random split would leak: two pairs sharing an S1
     share its features almost entirely, so val "accuracy" would mostly be
     memorization of train S1s, not generalization to unseen ones)
  4. compute pair features fresh from raw text
  5. train a LightGBM binary classifier (label = is_match)
  6. evaluate with the CHALLENGE'S OWN metric, not a generic classification
     one: per-S1-entity F0.5 (precision-weighted 2x), averaged (macro) over
     every val S1 including true singletons (empty-true + empty-pred = 1.0
     by definition; exactly one empty = 0.0). Sweep the decision threshold
     to find the one that actually maximizes THIS metric, not accuracy/AUC.

Usage:
    pip install rapidfuzz lightgbm
    python train_matcher.py --candidates cand_train.parquet \
        --data-dir dataset --translit-dict translit_dict.json \
        --out-model matcher.lgb --out-meta matcher_meta.json
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from normalize import clean_name, clean_address, numeric_suffix_overlap
from build_translit_dict import tokens as split_tokens, is_native


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False, quoting=3)


def latinize_ci(text: str, translit_dict: dict) -> str:
    if not is_native(text):
        return text
    out = []
    for raw in text.split():
        key = raw.casefold().strip(".,()[]{}'\"")
        out.append(translit_dict.get(key, raw))
    return " ".join(out)


# ---------------------------------------------------------------------------
# Entity feature index: built ONCE per entity_id actually referenced by the
# candidates parquet (S1 ids + every distinct candidate id), not over the
# full 10M-row S2/S3 pool -- most S2/S3 rows are never anyone's candidate.
# ---------------------------------------------------------------------------
def build_entity_index(df: pd.DataFrame, translit_dict: dict) -> dict:
    idx = {}
    for eid, name_raw, addr_raw in zip(df["entity_id"], df["business_name"], df["business_address"]):
        lat = latinize_ci(name_raw, translit_dict)
        ninfo = clean_name(lat)
        ainfo = clean_address(addr_raw)
        variants = [{
            "squashed": v["squashed"],
            "casefolded": v["casefolded"],
            "tokens": frozenset(split_tokens(v["no_suffix"])),
        } for v in ninfo["variants"]]
        idx[eid] = {
            "variants": variants,
            "is_domain_like": ninfo["is_domain_like"],
            "had_alias_marker": ninfo["had_alias_marker"],
            "had_phone_suffix": ninfo["had_phone_suffix"],
            "is_native_name": is_native(name_raw),
            "addr_tokens": ainfo["word_tokens"],
            "addr_numeric": ainfo["numeric_tokens"],
            "addr_is_empty": ainfo["is_empty"],
        }
    return idx


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def pair_features(s1_info: dict, cand_info: dict, agreement: int, idf: dict) -> dict:
    # best-of-all-variant-pairs so an f/k/a alias on either side doesn't
    # silently zero out a genuine match against the OTHER variant.
    best_ratio = best_tsr = best_tset = best_name_jac = 0.0
    best_squash_len_ratio = 0.0
    for va in s1_info["variants"]:
        for vb in cand_info["variants"]:
            best_ratio = max(best_ratio, fuzz.ratio(va["casefolded"], vb["casefolded"]))
            best_tsr = max(best_tsr, fuzz.token_sort_ratio(va["casefolded"], vb["casefolded"]))
            best_tset = max(best_tset, fuzz.token_set_ratio(va["casefolded"], vb["casefolded"]))
            best_name_jac = max(best_name_jac, jaccard(va["tokens"], vb["tokens"]))
            la, lb = len(va["squashed"]), len(vb["squashed"])
            if la or lb:
                best_squash_len_ratio = max(best_squash_len_ratio, min(la, lb) / max(la, lb, 1))

    shared_name_tokens = set()
    for va in s1_info["variants"]:
        for vb in cand_info["variants"]:
            shared_name_tokens |= (va["tokens"] & vb["tokens"])
    idf_shared = sum(idf.get(t, 0.0) for t in shared_name_tokens)

    addr_jac = jaccard(s1_info["addr_tokens"], cand_info["addr_tokens"])
    num_overlap = numeric_suffix_overlap(s1_info["addr_numeric"], cand_info["addr_numeric"])
    na, nb = len(s1_info["addr_tokens"]), len(cand_info["addr_tokens"])
    addr_len_ratio = (min(na, nb) / max(na, nb, 1)) if (na or nb) else 0.0

    return {
        "agreement_count": agreement,
        "name_ratio": best_ratio,
        "name_token_sort_ratio": best_tsr,
        "name_token_set_ratio": best_tset,
        "name_jaccard": best_name_jac,
        "name_len_ratio": best_squash_len_ratio,
        "idf_shared_name": idf_shared,
        "addr_jaccard": addr_jac,
        "addr_numeric_suffix_overlap": num_overlap,
        "addr_len_ratio": addr_len_ratio,
        "either_domain_like": int(s1_info["is_domain_like"] or cand_info["is_domain_like"]),
        "either_alias_marker": int(s1_info["had_alias_marker"] or cand_info["had_alias_marker"]),
        "either_phone_suffix": int(s1_info["had_phone_suffix"] or cand_info["had_phone_suffix"]),
        "either_addr_empty": int(s1_info["addr_is_empty"] or cand_info["addr_is_empty"]),
        "native_script_mismatch": int(s1_info["is_native_name"] != cand_info["is_native_name"]),
    }


FEATURE_NAMES = [
    "agreement_count", "name_ratio", "name_token_sort_ratio", "name_token_set_ratio",
    "name_jaccard", "name_len_ratio", "idf_shared_name", "addr_jaccard",
    "addr_numeric_suffix_overlap", "addr_len_ratio", "either_domain_like",
    "either_alias_marker", "either_phone_suffix", "either_addr_empty",
    "native_script_mismatch",
]


def entity_f_half(true_ids: set, pred_ids: set) -> float:
    """Challenge's own per-S1 metric: F0.5 = 1.25*P*R / (0.25*P + R)."""
    if not true_ids and not pred_ids:
        return 1.0
    if not true_ids or not pred_ids:
        return 0.0
    tp = len(true_ids & pred_ids)
    if tp == 0:
        return 0.0
    p = tp / len(pred_ids)
    r = tp / len(true_ids)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f_half_at_threshold(val_df: pd.DataFrame, threshold: float) -> float:
    scores = []
    for _, grp in val_df.groupby("source1_entity_id"):
        true_ids = set(grp.loc[grp["label"] == 1, "candidate_entity_id"])
        # true_ids here is "true matches THIS S1 has among val's candidate
        # rows" -- since every ground-truth pair for train S1s is present
        # as a labeled row whenever blocking retrieved it, and rows where
        # blocking missed a true match simply never appear as label=1
        # anywhere, so those misses still correctly fail to be predicted.
        pred_ids = set(grp.loc[grp["prob"] >= threshold, "candidate_entity_id"])
        scores.append(entity_f_half(true_ids, pred_ids))
    return float(np.mean(scores)) if scores else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="cand_train.parquet")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--split", default="train", choices=["train"])
    ap.add_argument("--translit-dict", default="translit_dict.json")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-model", default="matcher.lgb")
    ap.add_argument("--out-meta", default="matcher_meta.json")
    a = ap.parse_args()

    with open(a.translit_dict, encoding="utf-8") as f:
        translit_dict = json.load(f)

    log(f"loading candidates from {a.candidates} ...")
    cand = pd.read_parquet(a.candidates)
    d = os.path.join(a.data_dir, a.split)
    s1 = read_tsv(os.path.join(d, f"{a.split}_source1.tsv"))
    s2 = read_tsv(os.path.join(d, f"{a.split}_source2.tsv"))
    s3 = read_tsv(os.path.join(d, f"{a.split}_source3.tsv"))
    gt = read_tsv(os.path.join(d, f"{a.split}_ground_truth.tsv"))
    gt["true_ids"] = gt["matched_entity_ids"].map(lambda s: set(x.strip() for x in s.split(",") if x.strip()))
    gt = gt.set_index("source1_entity_id")["true_ids"]

    log("exploding candidates parquet into pairs ...")
    pairs = cand[["source1_entity_id", "candidate_entity_ids", "candidate_agreement"]].explode(
        ["candidate_entity_ids", "candidate_agreement"]
    ).rename(columns={"candidate_entity_ids": "candidate_entity_id", "candidate_agreement": "agreement_count"})
    pairs = pairs.dropna(subset=["candidate_entity_id"]).reset_index(drop=True)
    true_map = gt.to_dict()
    pairs["label"] = [
        int(cid in true_map.get(s1id, set()))
        for s1id, cid in zip(pairs["source1_entity_id"], pairs["candidate_entity_id"])
    ]
    log(f"{len(pairs):,} pairs, {pairs['label'].sum():,} positive ({pairs['label'].mean() * 100:.2f}%)")

    log("building entity feature index (only entities referenced as S1 or as a candidate) ...")
    needed_s1 = set(pairs["source1_entity_id"])
    needed_cand = set(pairs["candidate_entity_id"])
    s1_idx = build_entity_index(s1[s1["entity_id"].isin(needed_s1)], translit_dict)
    cand_idx = build_entity_index(s2[s2["entity_id"].isin(needed_cand)], translit_dict)
    cand_idx.update(build_entity_index(s3[s3["entity_id"].isin(needed_cand)], translit_dict))
    log(f"indexed {len(s1_idx):,} S1 + {len(cand_idx):,} candidate entities")

    log("computing a lightweight IDF table over referenced name tokens ...")
    all_infos = list(s1_idx.values()) + list(cand_idx.values())
    doc_freq = {}
    for info in all_infos:
        seen = set()
        for v in info["variants"]:
            seen |= v["tokens"]
        for t in seen:
            doc_freq[t] = doc_freq.get(t, 0) + 1
    n_docs = len(all_infos)
    idf = {t: np.log(1 + n_docs / (1 + df)) for t, df in doc_freq.items()}

    log(f"computing {len(pairs):,} pair features ...")
    feat_rows = []
    for s1id, cid, agree in zip(pairs["source1_entity_id"], pairs["candidate_entity_id"], pairs["agreement_count"]):
        feat_rows.append(pair_features(s1_idx[s1id], cand_idx[cid], int(agree), idf))
    X = pd.DataFrame(feat_rows, columns=FEATURE_NAMES)
    y = pairs["label"].to_numpy()

    log("entity-level train/val split ...")
    rng = np.random.default_rng(a.seed)
    uniq_s1 = np.array(sorted(needed_s1))
    rng.shuffle(uniq_s1)
    n_val = int(len(uniq_s1) * a.val_frac)
    val_s1 = set(uniq_s1[:n_val])
    is_val = pairs["source1_entity_id"].isin(val_s1).to_numpy()

    X_train, y_train = X[~is_val], y[~is_val]
    X_val, y_val = X[is_val], y[is_val]
    log(f"train pairs={len(X_train):,} (S1={len(uniq_s1) - n_val:,}) | "
        f"val pairs={len(X_val):,} (S1={n_val:,})")

    import lightgbm as lgb
    train_set = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_set)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "is_unbalance": True,
        "verbose": -1,
        "seed": a.seed,
    }
    log("training LightGBM ...")
    booster = lgb.train(
        params, train_set, num_boost_round=500,
        valid_sets=[val_set], valid_names=["val"],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
    )
    log(f"best iteration: {booster.best_iteration}, val AUC: {booster.best_score['val']['auc']:.4f}")

    val_df = pairs[is_val].copy()
    val_df["prob"] = booster.predict(X_val, num_iteration=booster.best_iteration)

    log("sweeping decision threshold against the CHALLENGE's macro F0.5 ...")
    best_t, best_f = 0.5, -1.0
    sweep = {}
    for t in np.arange(0.05, 0.96, 0.05):
        f = macro_f_half_at_threshold(val_df, t)
        sweep[round(float(t), 2)] = f
        if f > best_f:
            best_f, best_t = f, float(t)
    log(f"best threshold={best_t:.2f} -> macro F0.5={best_f:.4f} on val "
        f"(n_S1={n_val:,}, includes true singletons)")
    log(f"threshold sweep: {sweep}")

    booster.save_model(a.out_model)
    with open(a.out_meta, "w") as f:
        json.dump({
            "features": FEATURE_NAMES,
            "best_threshold": best_t,
            "val_macro_f_half": best_f,
            "val_auc": booster.best_score["val"]["auc"],
            "best_iteration": booster.best_iteration,
            "threshold_sweep": sweep,
            "feature_importance": dict(zip(FEATURE_NAMES, booster.feature_importance("gain").tolist())),
        }, f, indent=2)
    log(f"wrote {a.out_model} and {a.out_meta}")


if __name__ == "__main__":
    main()

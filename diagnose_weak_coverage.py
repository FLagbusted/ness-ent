"""
Diagnostic: of the S1 entities blocking is MISSING a true match for, how many
were flagged "weak" (and got the TF-IDF rescue) vs not? Answers directly from
files you already have -- no re-run needed. Same weak-flagging formula as
blocking_recall_v5.py's main(), recomputed here from n_before_prune and
candidate_agreement, both already saved in the candidates parquet.
"""
import argparse
import pandas as pd


def is_weak(n_before_prune, candidate_agreement, weak_threshold, min_confident_agreement):
    if n_before_prune < weak_threshold:
        return True
    return (candidate_agreement[0] if len(candidate_agreement) else 0) < min_confident_agreement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="cand_train.parquet")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--weak-threshold", type=int, default=10)
    ap.add_argument("--min-confident-agreement", type=int, default=2)
    a = ap.parse_args()

    cand = pd.read_parquet(a.candidates)
    gt = pd.read_csv(f"{a.data_dir}/{a.split}/{a.split}_ground_truth.tsv", sep="\t", dtype=str)
    gt["true_ids"] = gt["matched_entity_ids"].map(lambda s: set(x.strip() for x in str(s).split(",") if x.strip()))
    gt_map = dict(zip(gt["source1_entity_id"], gt["true_ids"]))

    rows = []
    for _, r in cand.iterrows():
        true_ids = gt_map.get(r["source1_entity_id"], set())
        if not true_ids:
            continue  # true singletons aren't a recall question
        found = bool(true_ids & set(r["candidate_entity_ids"]))
        weak = is_weak(r["n_before_prune"], r["candidate_agreement"], a.weak_threshold, a.min_confident_agreement)
        rows.append((found, weak))
    df = pd.DataFrame(rows, columns=["found", "weak"])

    missed = df[~df["found"]]
    print(f"total S1 with >=1 true match: {len(df):,}")
    print(f"missed (blocking recall failure): {len(missed):,} ({len(missed) / len(df) * 100:.2f}%)")
    if len(missed):
        caught = missed["weak"].sum()
        slipped = (~missed["weak"]).sum()
        print(f"  of those misses: {caught:,} ({caught / len(missed) * 100:.1f}%) WERE flagged weak "
              f"(TF-IDF ran but still didn't find it -- retrieval depth/k or corpus-side issue)")
        print(f"  of those misses: {slipped:,} ({slipped / len(missed) * 100:.1f}%) were NOT flagged weak "
              f"(TF-IDF never even ran on these -- selection-heuristic gap)")


if __name__ == "__main__":
    main()

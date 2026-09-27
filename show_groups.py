#!/usr/bin/env python3
"""
Print real examples from the TRAIN set so normalisation / blocking rules can be designed
from evidence instead of guesses. Read-only; needs only pandas.

    python show_groups.py --data-dir dataset --n 6 --seed 1 > groups.txt

Sections printed:
  A. India groups whose matched S2/S3 name is in a native (Indic) script
  B. India groups, Latin only
  C. US groups
  D. S1 entities with NO matches (singletons)
  E. Orphans: S2/S3 records that match no S1 (the ~26% decoys)
"""
import argparse
import os
import random

import pandas as pd

NATIVE = r"[\u0900-\u0D7F]"  # Devanagari .. Malayalam blocks (all Indic scripts seen in the census)


def read(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False, quoting=3)


def fmt(rid, row):
    return f"{rid} | {row.business_name} | {row.business_address} | {row.country}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--n", type=int, default=6, help="groups per section")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    tr = os.path.join(a.data_dir, "train")

    gt = read(os.path.join(tr, "train_ground_truth.tsv"))
    gt["ids"] = gt["matched_entity_ids"].map(lambda s: [x.strip() for x in s.split(",") if x.strip()])
    recs = pd.concat([read(os.path.join(tr, f"train_source{i}.tsv")) for i in (1, 2, 3)]).set_index("entity_id")
    native = set(recs.index[recs["business_name"].str.contains(NATIVE, regex=True)])

    matched = set(x for l in gt["ids"] for x in l)
    gt["country"] = gt["source1_entity_id"].map(recs["country"])
    gt["n"] = gt["ids"].map(len)
    gt["has_native"] = False
    # the native check is only done on a random sample of groups (enough to find examples)
    samp = gt.sample(min(len(gt), 400_000), random_state=a.seed).index
    gt.loc[samp, "has_native"] = gt.loc[samp, "ids"].map(lambda l: any(x in native for x in l))

    def pick(mask):
        idx = list(gt.index[mask])
        rnd.shuffle(idx)
        return gt.loc[idx[: a.n]]

    def show_groups(title, sub):
        print(f"\n{'=' * 8} {title} {'=' * 8}")
        for _, g in sub.iterrows():
            print(fmt(g.source1_entity_id, recs.loc[g.source1_entity_id]))
            for m in g.ids:
                print("     " + fmt(m, recs.loc[m]))
            print()

    show_groups("A. INDIA, matched name in native script", pick((gt.country == "India") & gt.has_native & (gt.n > 0)))
    show_groups("B. INDIA, Latin only", pick((gt.country == "India") & ~gt.has_native & (gt.n > 1)))
    show_groups("C. US", pick((gt.country == "US") & (gt.n > 1)))
    print(f"\n{'=' * 8} D. S1 with NO matches {'=' * 8}")
    for _, g in pick(gt.n == 0).iterrows():
        print(fmt(g.source1_entity_id, recs.loc[g.source1_entity_id]))
    print(f"\n{'=' * 8} E. ORPHANS: S2/S3 records that match no S1 {'=' * 8}")
    orphan_ids = [i for i in recs.index if i[:2] in ("S2", "S3") and i not in matched]
    for i in rnd.sample(orphan_ids, min(a.n * 2, len(orphan_ids))):
        print(fmt(i, recs.loc[i]))


if __name__ == "__main__":
    main()

"""
Train-vs-test shift check for the REVERSE features (80% of the model's gain).

  python density_check.py cand_train_us_R20.parquet cand_train_india_R20.parquet \
      cand_test_us_R20.parquet cand_test_india_R20.parquet cand_test_france_R20.parquet

The reverse columns count/score the OTHER S1 rows that list the same record. If the test split
holds fewer S1 entities per country than train (US: ~663k test vs ~1.32M train), each record has
fewer competing S1 rows at test time -> the model, trained on denser competition, over-accepts.
Prints, per file and country, for the top-5 candidates of every row:
  rev_n>0     share with at least one competing S1
  mean rev_n  average number of competing S1 rows
  beaten      share where a competing S1 has a HIGHER rank score (the case the model rejects)
If test's numbers are clearly lower than train's for US/India, the shift is real.
"""
import sys

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

COLS = ["country", "candidate_rank_score", "candidate_rev_n", "candidate_rev_rank_score"]


def main(paths, top=5):
    print(f"{'file':34s} {'country':8s} {'S1 rows':>9s} {'rev_n>0':>8s} {'mean rev_n':>10s} {'beaten':>7s}")
    for p in paths:
        acc = {}
        for b in pq.ParquetFile(p).iter_batches(batch_size=200_000, columns=COLS):
            c = b.column(0).to_numpy(zero_copy_only=False)
            ls = [b.column(i) for i in range(1, 4)]
            off = ls[0].offsets.to_numpy()
            k = np.minimum(np.diff(off), top)
            idx = np.repeat(off[:-1], k) + (np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k))
            r, n, ro = (x.values.to_numpy(zero_copy_only=False)[idx] for x in ls)
            cc = np.repeat(c, k)
            for cn in np.unique(c):
                m = cc == cn
                s = acc.setdefault(cn, [0, 0.0, 0.0, 0.0, 0])
                s[0] += m.sum()
                s[1] += (n[m] > 0).sum()
                s[2] += n[m].sum()
                s[3] += ((n[m] > 0) & (ro[m] > r[m])).sum()
                s[4] += int((c == cn).sum())
        for cn, (tot, pos, sn, beat, rows) in sorted(acc.items()):
            print(f"{p[-34:]:34s} {cn:8s} {rows:9,} {pos / tot * 100:7.1f}% {sn / tot:10.2f} {beat / tot * 100:6.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])

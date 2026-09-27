
"""
Re-apply a decision rule to saved pair scores -- minutes instead of re-running predict.

  python redecide.py --scores-dir output_v9 --out-dir output_v9b [--decision decision.json]
      [--owner-exclusive auto|on|off]

Reads <scores-dir>/scores.parquet (every pair with p > 0, written by predict_matches.py
--save-scores 1), scores_rows.parquet and scores_meta.json. With --decision (written by
tune_decision.py) it applies that rule and, if present, the probability calibration;
otherwise it re-applies the model's own rule (must reproduce predict's output exactly).
Writes <out-dir>/matching_results.tsv. candidate_pairs.tsv does not change: copy it from
<scores-dir>.
"""
import argparse
import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import er_features as ef


def calibrate(p, cal):
    if not cal:
        return p
    out = np.interp(p, np.asarray(cal["x"]), np.asarray(cal["y"]))
    return np.where(p > 0, out, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--decision", default="", help="decision.json from tune_decision.py (default: model's rule)")
    ap.add_argument("--owner-exclusive", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--country-rule", action="append", default=[],
                    help="per-country override on top of the rule, repeatable: 'France:empty_bias=0.8,m_extra=0.1' "
                         "or 'France:nocal' (skip calibration) or 'France:model' (the model's own rule, no calibration)")
    a = ap.parse_args()
    T0 = time.time()
    sm = json.load(open(os.path.join(a.scores_dir, "scores_meta.json")))
    rule, cal = sm["rule"], None
    if a.decision:
        d = json.load(open(a.decision))
        rule, cal = d["rule"], d.get("calibration")
        ef.log(f"decision from {a.decision}: {rule} calibration={'yes' if cal else 'no'}")
    else:
        ef.log(f"model's own rule: {rule}")
    oe = {"on": True, "off": False}.get(a.owner_exclusive, sm["owner_exclusive"])

    rows_t = pq.read_table(os.path.join(a.scores_dir, "scores_rows.parquet"))
    n_s1 = rows_t.num_rows
    s1_ids = rows_t["s1_id"].to_pylist()
    t = pq.read_table(os.path.join(a.scores_dir, "scores.parquet"))
    row = t["row"].to_numpy().astype(np.int64)
    d = pc.dictionary_encode(t["cand"]).combine_chunks()
    key = d.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    cand_dict = d.dictionary
    p_raw = t["p"].to_numpy().astype(np.float64)
    del t
    ctry = np.array(rows_t["country"].to_pylist())
    over = {}
    for spec in a.country_rule:
        cn, kv = spec.split(":", 1)
        o_ = over.setdefault(cn, {"rule": dict(rule), "cal": cal})
        for item in kv.split(","):
            if item == "nocal":
                o_["cal"] = None
            elif item == "model":
                o_["rule"], o_["cal"] = dict(sm["rule"]), None
            else:
                k_, v_ = item.split("=")
                o_["rule"][k_] = float(v_)
    for cn, o_ in over.items():
        if cn not in set(ctry):
            raise SystemExit(f"--country-rule: no rows for country {cn!r}")
        ef.log(f"override {cn}: {o_['rule']} calibration={'yes' if o_['cal'] else 'no'}")
    p = calibrate(p_raw, cal)
    ef.log(f"{len(p):,} scored pairs for {n_s1:,} S1 rows ({time.time() - T0:.0f}s)")

    o = np.argsort(row, kind="stable")          # rules expect pairs grouped by row
    row, key, p, p_raw = row[o], key[o], p[o], p_raw[o]
    pr_ctry = ctry[row]
    pred = np.zeros(len(p), bool)
    base_m = ~np.isin(pr_ctry, list(over)) if over else np.ones(len(p), bool)
    groups = [(base_m, rule)]
    for cn, o_ in over.items():
        m = pr_ctry == cn
        p[m] = calibrate(p_raw[m], o_["cal"])       # the override's own calibration (or none)
        groups.append((m, o_["rule"]))
    for m, rl in groups:
        pm = ef.apply_rule(rl, row[m], p[m], key[m], n_s1, oe_local=False)
        if rl.get("min_p"):
            pm &= p[m] >= rl["min_p"]
        pred[m] = pm
    r, k, pp = row[pred], key[pred], p[pred]
    if oe and len(pp):
        keep = ef.owner_exclusive(np.ones(len(pp), bool), pp, k)
        ef.log(f"owner-exclusive removed {int((~keep).sum()):,} of {len(pp):,} accepted pairs")
        r, k, pp = r[keep], k[keep], pp[keep]
    order = np.lexsort((-pp, r))
    r, k = r[order], k[order]
    mids = cand_dict.take(pa.array(k)).to_pylist()
    npred = np.bincount(r, minlength=n_s1)
    starts = np.concatenate([[0], np.cumsum(npred)[:-1]])
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "matching_results.tsv"), "w") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for i, sid in enumerate(s1_ids):
            fm.write(f"{sid}\t{','.join(mids[starts[i]:starts[i] + npred[i]])}\n")
        for sid in sm["absent_s1"]:
            fm.write(f"{sid}\t\n")
    for cn in sorted(set(ctry)):
        m = ctry == cn
        ef.log(f"  {cn:8s}: {m.sum():9,} rows | empty-rate {(npred[m] == 0).mean() * 100:5.1f}% | "
               f"avg matches/row {npred[m].mean():.2f}")
    ef.log(f"wrote {a.out_dir}/matching_results.tsv in {time.time() - T0:.0f}s "
           f"(copy candidate_pairs.tsv from {a.scores_dir})")


if __name__ == "__main__":
    main()

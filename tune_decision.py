"""
Post-stage-2 decision tuning with HONEST (2-fold, by S1 row) cross-validation.

  python tune_decision.py --candidates cand_train_us_R20.parquet,cand_train_india_R20.parquet \
      --data-dir dataset --translit-dict translit_dict.json --model-dir matcher_v9 \
      --max-s1 400000 --val-rows 20000 --workers 12 --out decision.json

Use the SAME --candidates / --max-s1 / --val-rows / --seed as the training run: the script
rebuilds exactly the training run's validation rows and features (same record set, so the
same IDF / name-frequency values), scores them with the saved models (seed ensemble + stage 2)
and checks it reproduces meta.json's val F0.5.

Then, per method, parameters are picked on one half of the validation rows and scored on the
other half (and vice versa):
  thresholds         t_top / t_extra (+ owner-exclusive)          -- the current rule family
  ef                 expected-F0.5 rule, training's coarse grid   -- the current rule family
  ef_fine            expected-F0.5 on a fine grid
  iso+ef_fine        isotonic probability calibration, then ef_fine
  ef_fine+min_p      ef_fine, then drop accepted pairs below a floor
  iso+ef_fine+min_p  both
Also printed: the ORACLE ceiling (best possible top-k per row given these probabilities'
ranking) -- the most any decision rule could still gain.
Writes decision.json (winner re-fit on ALL validation rows) for redecide.py, and
val_scores.npz for further offline analysis.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import er_features as ef
from er_features import row_f_half

EF_M_FINE = [round(float(x), 3) for x in np.arange(0.0, 1.01, 0.05)]
EF_B_FINE = [round(float(x), 3) for x in np.geomspace(0.3, 20, 24)]
EF_M_COARSE = [0.0, 0.03, 0.1, 0.2, 0.4, 0.7]
EF_B_COARSE = [0.15, 0.25, 0.35, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0, 4.5, 7.0, 12.0]
THRESH = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]
MIN_P = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]


def macro(row, pred, y, nt, n):
    tp = np.bincount(row, weights=pred & y, minlength=n)
    npred = np.bincount(row, weights=pred, minlength=n)
    return row_f_half(tp, npred, nt)


def fit_iso(p, y):
    from sklearn.isotonic import IsotonicRegression
    m = p > 0
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p[m], y[m].astype(float))
    return {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}


def apply_cal(p, cal):
    if cal is None:
        return p
    return np.where(p > 0, np.interp(p, cal["x"], cal["y"]), 0.0)


def candidates_for(method):
    """-> list of (params dict) for a method."""
    out = []
    if method == "thresholds":
        for oe in (False, True):
            for tt in THRESH:
                for tx in THRESH:
                    out.append({"kind": "thresh", "t_top": tt, "t_extra": tx, "owner_exclusive": oe})
    else:
        ms, bs = (EF_M_COARSE, EF_B_COARSE) if method == "ef" else (EF_M_FINE, EF_B_FINE)
        mins = MIN_P if "min_p" in method else [0.0]
        for m in ms:
            for b in bs:
                for mp in mins:
                    out.append({"kind": "ef", "m_extra": m, "empty_bias": b, "owner_exclusive": False, "min_p": mp})
    return out


def decide(params, row, p, key, n):
    pred = ef.apply_rule(params, row, p, key, n, oe_local=None)
    if params.get("min_p"):
        pred &= p >= params["min_p"]
    return pred


def best_params(method, row, p, y, key, nt, n):
    """grid on (row, p...) -> (best params, best macro); EF variants share one sort per m."""
    best, bv = None, -1
    if method == "thresholds":
        dec = ef.Decider(row, p, key, n)
        for prm in candidates_for(method):
            pr = dec(prm["t_top"], prm["t_extra"], prm["owner_exclusive"])
            v = macro(row, pr, y, nt, n).mean()
            if v > bv:
                best, bv = prm, v
        return best, bv
    efd = ef.EFDecider(row, p, n)
    for prm in candidates_for(method):
        pr = efd(prm["m_extra"], prm["empty_bias"])
        if prm["min_p"]:
            pr = pr & (p >= prm["min_p"])
        v = macro(row, pr, y, nt, n).mean()
        if v > bv:
            best, bv = prm, v
    return best, bv


def subset(mask_rows, row, *arrs):
    """pairs of the selected S1 rows, re-indexed 0..k-1."""
    keep = mask_rows[row]
    remap = -np.ones(len(mask_rows), np.int64)
    remap[mask_rows] = np.arange(mask_rows.sum())
    return (remap[row[keep]],) + tuple(x[keep] for x in arrs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--translit-dict", default="translit_dict.json")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--max-s1", type=int, default=100_000)
    ap.add_argument("--topk-use", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--val-rows", type=int, default=0)
    ap.add_argument("--val-country", default="")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", default="decision.json")
    ap.add_argument("--exclude-seen", default="",
                    help="candidate files the MODEL was trained on (comma-sep); their --seen-max-s1/--seen-seed "
                         "sample is re-drawn and those S1 ids are removed from validation (use when scoring a "
                         "different candidates file, e.g. the density-matched D parquets)")
    ap.add_argument("--seen-max-s1", type=int, default=0)
    ap.add_argument("--seen-seed", type=int, default=1)
    ap.add_argument("--compare-decision", default="", help="also score this decision.json (e.g. the v9d one) here")
    a = ap.parse_args()
    T0 = time.time()
    import lightgbm as lgb
    meta = json.load(open(os.path.join(a.model_dir, "meta.json")))
    if meta["features"] != ef.FEATURES:
        raise SystemExit("feature list differs from training")
    tau0 = meta["tau0"]
    feat_idx = np.array(meta["feat_idx"])

    # ---------------- rebuild the training run's validation rows (same RNG sequence as train_matcher.py)
    rng = np.random.RandomState(a.seed)
    cand, n_all, _ = ef.load_candidates(a.candidates.split(","), a.max_s1, rng)
    nr = cand.num_rows
    s1_ids = cand["source1_entity_id"].combine_chunks()
    country = cand["country"].to_numpy(zero_copy_only=False)
    gt = pd.read_csv(os.path.join(a.data_dir, "train", "train_ground_truth.tsv"), sep="\t", dtype=str,
                     keep_default_na=False, na_filter=False, quoting=3).set_index("source1_entity_id")
    truth_lists = [[x.strip() for x in s.split(",") if x.strip()]
                   for s in gt["matched_entity_ids"].reindex(s1_ids.to_pylist()).fillna("")]
    del gt
    if a.val_country:
        val_rows = country == a.val_country
    else:
        perm = rng.permutation(nr)
        val_rows = np.zeros(nr, bool)
        val_rows[perm[:(min(a.val_rows, nr // 2) if a.val_rows else int(nr * a.val_frac))]] = True
    if a.exclude_seen:
        paths = a.exclude_seen.split(",")
        ids_all = pa.concat_arrays([pq.read_table(p_, columns=["source1_entity_id"])["source1_entity_id"]
                                    .combine_chunks() for p_ in paths])
        n_seen_all = len(ids_all)
        if a.seen_max_s1 and a.seen_max_s1 < n_seen_all:
            pick = np.sort(np.random.RandomState(a.seen_seed).choice(n_seen_all, a.seen_max_s1, replace=False))
            ids_all = ids_all.take(pa.array(pick))
        seen = pc.is_in(s1_ids, value_set=pc.unique(ids_all)).to_numpy(zero_copy_only=False)
        ef.log(f"exclude-seen: {int((val_rows & seen).sum()):,} of {int(val_rows.sum()):,} val rows were in the "
               f"model's training sample -> removed")
        val_rows &= ~seen
        del ids_all
    vrows = np.flatnonzero(val_rows)
    nv = len(vrows)
    ef.log(f"validation rows rebuilt: {nv:,} of {nr:,}")

    # same record set as training -> identical IDF / name-frequency values
    lists_all = cand["candidate_entity_ids"].combine_chunks()
    if a.topk_use:
        lists_all = pc.list_slice(lists_all, 0, a.topk_use)
    truth_flat_all = pa.array([x for t in truth_lists for x in t], type=pa.string())
    needed = pa.concat_arrays([s1_ids, pc.unique(pc.list_flatten(lists_all)), truth_flat_all])
    del lists_all
    with open(a.translit_dict, encoding="utf-8") as f:
        translit = json.load(f)
    rec = ef.normalize_records(ef.read_records(a.data_dir, "train", needed), translit, a.workers)
    rec = ef.attach_name_freq(rec, a.data_dir, "train", translit, a.workers)
    ef.log(f"records ready ({time.time() - T0:.0f}s)")

    cv = cand.take(pa.array(vrows))
    P = ef.explode_candidates(cv, a.topk_use or None)
    s1_rec = ef.rec_index(rec, cv["source1_entity_id"].combine_chunks())
    X, b_raw = ef.build_features(cv, P, rec, s1_rec, a.workers)
    X1 = X if len(feat_idx) == X.shape[1] else X[:, feat_idx]
    row = P["row"]
    vt = [truth_lists[i] for i in vrows]
    nt = np.array([len(t) for t in vt], np.float64)
    tr = ef.rec_index(rec, pa.array([x for t in vt for x in t], type=pa.string()))
    trow = np.repeat(np.arange(nv), nt.astype(np.int64))
    R = len(rec["ids"]) + 1
    tkeys = np.sort(trow * R + tr)
    y = np.isin(row * R + np.maximum(b_raw, -1), tkeys) & (b_raw >= 0)
    key = np.where(b_raw < 0, -2 - np.arange(len(b_raw)), b_raw)
    is_s3 = X[:, ef.FEATURES.index("is_s3")] > 0.5

    # ---------------- score with the saved pipeline
    t = time.time()
    models = [lgb.Booster(model_file=os.path.join(a.model_dir, "model.txt"))] + \
             [lgb.Booster(model_file=os.path.join(a.model_dir, f)) for f in meta.get("stage1_extra", [])]
    p = ef.floor_p(np.mean([m.predict(X1, num_threads=a.workers) for m in models], axis=0), tau0)
    if meta.get("stage2"):
        b2 = lgb.Booster(model_file=os.path.join(a.model_dir, "model2.txt"))
        sel, F2 = ef.consensus_features(row, p, b_raw, is_s3, rec, nv, a.workers, tau0)
        if len(sel):
            p = p.copy()
            p[sel] = ef.floor_p(b2.predict(ef.stage2_matrix(X1, sel, F2), num_threads=a.workers), tau0)
    del X, X1
    ef.log(f"scored {len(p):,} val pairs with {len(models)} stage-1 model(s) + stage 2 in {time.time() - t:.0f}s")
    rule0 = meta["rule"]
    base = macro(row, ef.apply_rule(rule0, row, p, key, nv, oe_local=None), y, nt, nv)
    ef.log(f"REPRODUCTION CHECK: val macro F0.5 with the model's rule = {base.mean():.4f} "
           f"(training reported {meta['val_macro_f05']:.4f})")
    if a.exclude_seen:
        ef.log("  (different candidates/rows than training -- this is the model's score in THAT world, not a replay)")
    elif abs(base.mean() - meta["val_macro_f05"]) > 0.002:
        ef.log("  !! does not match -- are --candidates/--max-s1/--val-rows/--seed the same as training?")
    vc = country[vrows]
    pred0 = ef.apply_rule(rule0, row, p, key, nv, oe_local=None)
    npred0 = np.bincount(row[pred0], minlength=nv)
    cmp_v = None
    if a.compare_decision:
        dcmp = json.load(open(a.compare_decision))
        pc_ = apply_cal(p, dcmp.get("calibration"))
        predc = decide(dcmp["rule"], row, pc_, key, nv)
        cmp_v = macro(row, predc, y, nt, nv)
        npredc = np.bincount(row[predc], minlength=nv)
        ef.log(f"COMPARE {a.compare_decision}: {cmp_v.mean():.4f} vs model's rule {base.mean():.4f}")
    for cn in sorted(set(vc)):
        m = vc == cn
        msg = (f"  {cn:8s}: {m.sum():7,} rows | model rule F0.5 {base[m].mean():.4f} | truth: empty "
               f"{(nt[m] == 0).mean() * 100:5.1f}% avg {nt[m].mean():.2f}/row | predicted: empty "
               f"{(npred0[m] == 0).mean() * 100:5.1f}% avg {npred0[m].mean():.2f}/row")
        if cmp_v is not None:
            msg += (f" || compare F0.5 {cmp_v[m].mean():.4f} empty {(npredc[m] == 0).mean() * 100:5.1f}% "
                    f"avg {npredc[m].mean():.2f}/row")
        ef.log(msg)
    np.savez_compressed(os.path.splitext(a.out)[0] + "_val_scores.npz", row=row, p=p, y=y, key=key, nt=nt)

    # only pairs with p > 0 can ever be accepted; zero pairs add nothing to EF sums (floored p)
    nz = p > 0
    row, p, y, key = row[nz], p[nz], y[nz], key[nz]

    # ---------------- oracle: best top-k per row by this ranking
    o = np.lexsort((-p, row))
    rs, ys = row[o], y[o]
    cnt = np.bincount(rs, minlength=nv)
    st = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    k = np.arange(len(rs)) - st[rs] + 1
    ctp = np.cumsum(ys) - np.concatenate([[0], np.cumsum(ys)])[st][rs]
    Fk = 1.25 * ctp / (0.25 * nt[rs] + k)
    bestF = np.where(nt == 0, 1.0, 0.0)
    has = cnt > 0
    bestF[has] = np.maximum(bestF[has], np.maximum.reduceat(Fk, st[has]))
    ef.log(f"ORACLE (uses the labels -- NOT reachable, an upper bound): best cut-off per row with these "
           f"probabilities = {bestF.mean():.4f} vs current rule {base.mean():.4f} "
           f"({(bestF.mean() - base.mean()) * 100:.2f} pts of loss sit in the ordering-is-right-but-cut-is-wrong zone)")

    # ---------------- 2-fold CV by S1 row
    rs2 = np.random.RandomState(7)
    fold = rs2.rand(nv) < 0.5
    methods = ["thresholds", "ef", "ef_fine", "iso+ef_fine", "ef_fine+min_p", "iso+ef_fine+min_p"]
    results = {}
    for mth in methods:
        t = time.time()
        tot, chosen = 0.0, []
        for fa in (fold, ~fold):
            ra, pa_, ya, ka = subset(fa, row, p, y, key)
            rb, pb, yb, kb = subset(~fa, row, p, y, key)
            nta, ntb = nt[fa], nt[~fa]
            cal = fit_iso(pa_, ya) if mth.startswith("iso") else None
            prm, _ = best_params(mth.replace("iso+", ""), ra, apply_cal(pa_, cal), ya, ka, nta, int(fa.sum()))
            pbc = apply_cal(pb, cal)
            vb = macro(rb, decide(prm, rb, pbc, kb, int((~fa).sum())), yb, ntb, int((~fa).sum()))
            tot += vb.sum()
            chosen.append(prm)
        results[mth] = tot / nv
        ef.log(f"  CV {mth:18s}: {results[mth]:.4f}   (params picked per half: {chosen[0]} | {chosen[1]}; "
               f"{time.time() - t:.0f}s)")
    ref = max(results["thresholds"], results["ef"])
    win = max(results, key=results.get)
    ef.log(f"best method: {win}  CV {results[win]:.4f} vs current rule families {ref:.4f} "
           f"-> {(results[win] - ref) * 100:+.3f} pts")

    # ---------------- re-fit the winner on ALL validation rows and write decision.json
    cal = fit_iso(p, y) if win.startswith("iso") else None
    prm, v_all = best_params(win.replace("iso+", ""), row, apply_cal(p, cal), y, key, nt, nv)
    if win in ("thresholds", "ef") or results[win] <= ref:
        ef.log("no method beats the current rule by CV -- decision.json keeps the model's own rule")
        prm, cal = rule0, None
    dec = {"rule": prm, "calibration": cal, "method": win, "cv": results,
           "val_in_sample": v_all, "model_dir": a.model_dir}
    with open(a.out, "w") as f:
        json.dump(dec, f, indent=1)
    ef.log(f"wrote {a.out}: rule={prm} calibration={'isotonic' if cal else 'none'}  (total {time.time() - T0:.0f}s)")
    ef.log(f"apply to the test scores:  python redecide.py --scores-dir output_v9 --out-dir output_v9d --decision {a.out}")


if __name__ == "__main__":
    main()

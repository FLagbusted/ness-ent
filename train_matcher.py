"""
Matcher v3 = v2 (LightGBM + F0.5 loss ledger) plus:

  STAGE 2 (--stage2 1, default): a second LightGBM on top of stage-1 probabilities.
     For every pair with p1 >= --tau0 it adds the row's probability landscape (max,
     second, sum, count >= 0.5, rank by p) and the candidate's similarity to the row's
     top-probability "anchor" records (true S2/S3 matches of one S1 are copies of each
     other). Targets the two biggest real-data losses: partial_recall_matcher and
     partial_extra_fp. Stage-1 probabilities for TRAIN rows are out-of-fold
     (--oof-folds), so stage 2 never learns from overfit in-sample p1.
  DECISION RULE (--rule auto): grid of the two-threshold rule AND an expected-F0.5 rule
     (per row, predict the top-k that maximizes expected F0.5 under the model's
     probabilities, or nothing). Best on validation wins and is saved in meta.json.
  --val-country X: validate on country X, train on the others (leave-one-country-out).
     Simulates an unseen country such as France.
  --drop-groups a,b: train without those feature groups (e.g. a country-agnostic model
     without "blocking" / "row"); combine with --val-country to see what transfers.

  python train_matcher.py --candidates cand_train.parquet --data-dir dataset \
      --translit-dict translit_dict.json --max-s1 60000 --lr 0.08 --workers 12 --out-dir matcher_v3

Writes model.txt (+ model2.txt for stage 2), meta.json, report.txt, loss_examples.txt.
predict_matches.py reads meta.json and applies whatever was chosen.

NOTE on owner-exclusive: validation holds a SAMPLE of S1 rows, so the S1 rows that
compete for the same S2/S3 record are mostly absent and owner-exclusive looks like a
no-op here. On the full test set it is not (see predict_matches.py contention report).
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

import er_features as ef
from er_features import FEATURES, FEATURE_GROUPS, CONS_FEATS, Decider, row_f_half

THRESH = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97,
          0.98, 0.99, 0.995, 0.998, 0.999, 0.9995]
EF_M = [0.0, 0.03, 0.1, 0.2, 0.4, 0.7]
EF_B = [0.15, 0.25, 0.35, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0, 4.5, 7.0, 12.0]

_REPORT = []


def out(msg=""):
    ef.log(msg)
    _REPORT.append(msg)


def auc(y, p):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    r = pd.Series(p).rank().to_numpy()
    npos = y.sum()
    return (r[y].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))


def score_pred(row, pred, y, nt, n_rows):
    tp = np.bincount(row, weights=pred & y, minlength=n_rows)
    npred = np.bincount(row, weights=pred, minlength=n_rows)
    return row_f_half(tp, npred, nt), tp, npred


def grid_search(row, p, y, key, nt, n_rows, rules=("thresh", "ef"), grid=THRESH):
    """-> best rule dict, {rule_tuple: macro}"""
    res = {}
    if "thresh" in rules:
        dec = Decider(row, p, key, n_rows)
        for oe in (False, True):
            for tt in grid:
                for tx in grid:
                    res[("thresh", tt, tx, oe)] = score_pred(row, dec(tt, tx, oe), y, nt, n_rows)[0].mean()
    if "ef" in rules:
        efd = ef.EFDecider(row, p, n_rows)
        for m in EF_M:
            for b in EF_B:
                pr = efd(m, b)
                res[("ef", m, b, False)] = score_pred(row, pr, y, nt, n_rows)[0].mean()
                res[("ef", m, b, True)] = score_pred(row, ef.owner_exclusive(pr, p, key), y, nt, n_rows)[0].mean()
    best = max(res, key=res.get)
    return to_rule(best), res


def to_rule(k):
    if k[0] == "thresh":
        return {"kind": "thresh", "t_top": k[1], "t_extra": k[2], "owner_exclusive": bool(k[3])}
    return {"kind": "ef", "m_extra": k[1], "empty_bias": k[2], "owner_exclusive": bool(k[3])}


def rule_str(r):
    if r["kind"] == "thresh":
        return f"thresholds t_top={r['t_top']} t_extra={r['t_extra']} owner_exclusive={r['owner_exclusive']}"
    return f"expected-F0.5 m_extra={r['m_extra']} empty_bias={r['empty_bias']} owner_exclusive={r['owner_exclusive']}"


def lgb_params(a, seed=None, over=None):
    p = dict(objective="binary", learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=100,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                metric="binary_logloss", num_threads=a.workers, verbose=-1,
                seed=a.seed if seed is None else seed)
    p.update(over or {})
    return p


def train_lgb(Xtr, ytr, Xva, yva, names, a, rounds=None, early=True, seed=None, wtr=None, over=None):
    """Early stopping on val logloss. With early=False no validation set is evaluated at all
    (OOF folds / ablation use a fixed round count, so per-iteration val metrics are wasted time)."""
    import lightgbm as lgb
    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, feature_name=names, free_raw_data=True)
    if not early:
        return lgb.train(lgb_params(a, seed, over), dtr, num_boost_round=rounds or a.rounds)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr, free_raw_data=True)
    cb = [lgb.log_evaluation(200), lgb.early_stopping(100, verbose=False)]
    return lgb.train(lgb_params(a, seed), dtr, num_boost_round=rounds or a.rounds, valid_sets=[dva],
                     valid_names=["val"], callbacks=cb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="v7 parquet(s), comma-separated")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--translit-dict", default="translit_dict.json")
    ap.add_argument("--max-s1", type=int, default=100_000, help="S1 rows used (memory ~ max_s1*topk*62*4 bytes)")
    ap.add_argument("--topk-use", type=int, default=0, help="use only the first k candidates per row (0 = all)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--val-country", default="", help="validate on this country only, train on the rest")
    ap.add_argument("--drop-groups", default="", help="feature groups to leave out, comma-separated")
    ap.add_argument("--val-rows", type=int, default=0, help="exact number of val S1 rows (overrides --val-frac)")
    ap.add_argument("--neg-frac", type=float, default=1.0,
                    help="train rows: keep this fraction of negatives ranked >= --neg-hard-k (weighted 1/frac)")
    ap.add_argument("--neg-hard-k", type=int, default=40, help="train rows: always keep negatives ranked below this")
    ap.add_argument("--chunk-rows", type=int, default=25_000, help="S1 rows per feature-building chunk")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=127)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--stage2", type=int, default=1)
    ap.add_argument("--n-models", type=int, default=1,
                    help="stage-1 seed ensemble: train N models (different seeds), average their probabilities; "
                         "kept only if validation improves")
    ap.add_argument("--oof-folds", type=int, default=3)
    ap.add_argument("--tau0", type=float, default=0.002, help="stage 2 scores only pairs with p1 >= tau0")
    ap.add_argument("--rule", default="auto", choices=["auto", "thresh", "ef"])
    ap.add_argument("--prefilter", type=int, default=1,
                    help="1 = fit a small cascade model so predict runs the big model only on plausible pairs "
                         "(kept only if the full predict path replayed on validation gives the same F0.5)")
    ap.add_argument("--ablation", type=int, default=0, help="1 = retrain stage 1 without each feature group")
    ap.add_argument("--ablation-frac", type=float, default=0.3)
    ap.add_argument("--examples", type=int, default=25)
    ap.add_argument("--out-dir", default="matcher_v3")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    T0 = time.time()
    rng = np.random.RandomState(a.seed)
    rules = ("thresh", "ef") if a.rule == "auto" else (a.rule,)

    drop = [g.strip() for g in a.drop_groups.split(",") if g.strip()]
    for g in drop:
        if g not in FEATURE_GROUPS:
            raise SystemExit(f"--drop-groups: unknown group {g!r}; choose from {list(FEATURE_GROUPS)}")
    feat_idx = np.array([i for i, f in enumerate(FEATURES)
                         if not any(f in FEATURE_GROUPS[g] for g in drop)])
    names1 = [FEATURES[i] for i in feat_idx]
    names2 = names1 + CONS_FEATS

    # ------------------------------------------------------------------ load
    cand, n_all, dup = ef.load_candidates(a.candidates.split(","), a.max_s1, rng)
    nr = cand.num_rows
    s1_ids = cand["source1_entity_id"].combine_chunks()
    out(f"candidates: {n_all:,} S1 rows in parquet, using {nr:,}" + (f" | dropped groups: {drop}" if drop else ""))
    country = (cand["country"].to_numpy(zero_copy_only=False) if "country" in cand.column_names
               else np.array(["?"] * nr))

    gt = pd.read_csv(os.path.join(a.data_dir, "train", "train_ground_truth.tsv"), sep="\t", dtype=str,
                     keep_default_na=False, na_filter=False, quoting=3).set_index("source1_entity_id")
    truth_lists = [[x.strip() for x in s.split(",") if x.strip()]
                   for s in gt["matched_entity_ids"].reindex(s1_ids.to_pylist()).fillna("")]
    s1_without_gt = int(gt["matched_entity_ids"].reindex(s1_ids.to_pylist()).isna().sum())
    # every S2/S3 id that has an owner in the FULL ground truth (owner-exclusive check + FP diagnosis)
    owned = pa.array([x.strip() for s in gt["matched_entity_ids"] for x in s.split(",") if x.strip()],
                     type=pa.string())
    multi_owner = len(owned) - len(pc.unique(owned))
    del gt
    nt = np.array([len(t) for t in truth_lists], np.float64)
    truth_flat = pa.array([x for t in truth_lists for x in t], type=pa.string())
    truth_row = np.repeat(np.arange(nr), nt.astype(np.int64))

    # ------------------------------------------------------------------ split (by S1 row, before features)
    if a.val_country:
        val_rows = country == a.val_country
        if not val_rows.any() or val_rows.all():
            raise SystemExit(f"--val-country {a.val_country}: need rows of it AND of other countries; "
                             f"have {sorted(set(country))}")
    else:
        perm = rng.permutation(nr)
        val_rows = np.zeros(nr, bool)
        val_rows[perm[:(min(a.val_rows, nr // 2) if a.val_rows else int(nr * a.val_frac))]] = True

    # ------------------------------------------------------------------ records
    lists_all = cand["candidate_entity_ids"].combine_chunks()
    if a.topk_use:
        lists_all = pc.list_slice(lists_all, 0, a.topk_use)
    row_lens = pc.list_value_length(lists_all).to_numpy(zero_copy_only=False).astype(np.int64)
    needed = pa.concat_arrays([s1_ids, pc.unique(pc.list_flatten(lists_all)), truth_flat])
    del lists_all
    t = time.time()
    with open(a.translit_dict, encoding="utf-8") as f:
        translit = json.load(f)
    recs_tbl = ef.read_records(a.data_dir, "train", needed)
    del needed
    rec = ef.normalize_records(recs_tbl, translit, a.workers)
    del recs_tbl
    rec = ef.attach_name_freq(rec, a.data_dir, "train", translit, a.workers)
    out(f"records: {len(rec['ids']):,} normalized in {time.time() - t:.0f}s")
    s1_rec = ef.rec_index(rec, s1_ids)
    truth_rec = ef.rec_index(rec, truth_flat)
    R = len(rec["ids"]) + 1
    tkeys = np.sort(truth_row.astype(np.int64) * R + truth_rec)

    # ------------------------------------------------------------------ features, chunk by chunk
    # Train rows keep: every positive, every negative ranked < --neg-hard-k, and a random
    # --neg-frac of the other negatives with weight 1/neg-frac (so the loss, and calibration,
    # match training on all pairs). Val rows keep everything, so every metric stays exact.
    t = time.time()
    Xs_, ys_, ws_, rows_, ranks_, bs_ = [], [], [], [], [], []
    tc = np.zeros(nr)
    n_seen, rows_dup = 0, 0
    for s0 in range(0, nr, a.chunk_rows):
        c = cand.slice(s0, a.chunk_rows)
        P = ef.explode_candidates(c, a.topk_use or None)
        Xc, bc = ef.build_features(c, P, rec, s1_rec[s0:s0 + c.num_rows], a.workers)
        g = P["row"] + s0
        pk = g * R + np.maximum(bc, -1)
        yc = np.isin(pk, tkeys) & (bc >= 0)
        rows_dup += len(pk) - len(np.unique(pk))
        tc += np.bincount(g, weights=yc, minlength=nr)
        n_seen += len(g)
        vr_c = val_rows[g]
        keep = vr_c | yc | (P["rank"] < a.neg_hard_k)
        wc = np.ones(len(g), np.float32)
        if a.neg_frac < 1.0:
            rnd = ~keep & (rng.rand(len(g)) < a.neg_frac)
            wc[rnd] = 1.0 / a.neg_frac
            keep |= rnd
        else:
            keep[:] = True
        Xs_.append(Xc[keep]); ys_.append(yc[keep]); ws_.append(wc[keep])
        rows_.append(g[keep]); ranks_.append(P["rank"][keep]); bs_.append(bc[keep])
        del Xc, P, c
    X = np.concatenate(Xs_); del Xs_
    y = np.concatenate(ys_); wt = np.concatenate(ws_); row = np.concatenate(rows_)
    rank = np.concatenate(ranks_); b_raw = np.concatenate(bs_)
    del ys_, ws_, rows_, ranks_, bs_
    npairs = len(row)
    out(f"features: {n_seen:,} pairs built, {npairs:,} kept ({npairs / max(n_seen, 1) * 100:.0f}%, "
        f"{X.nbytes / 2 ** 30:.1f} GB) x {X.shape[1]} in {time.time() - t:.0f}s")
    is_s3 = X[:, FEATURES.index("is_s3")] > 0.5
    b_idx = np.where(b_raw < 0, -2 - np.arange(len(b_raw)), b_raw)   # unique keys for missing ids

    # ------------------------------------------------------------------ sanity checks
    out("\n==================== SANITY CHECKS ====================")
    warn = []
    out(f"pairs kept {npairs:,} | positives {int(y.sum()):,} | S1 rows {nr:,} "
        f"(with >=1 true match: {(nt > 0).mean() * 100:.1f}%) | avg true matches/S1 {nt.mean():.2f}")
    out(f"ground truth: S2/S3 ids listed under more than one S1: {multi_owner:,} "
        f"({'owner-exclusive assumption holds' if multi_owner == 0 else 'owner-exclusive assumption is VIOLATED'})")
    if dup:
        warn.append(f"{dup:,} duplicate source1_entity_id rows in the candidates parquet(s)")
    if s1_without_gt:
        warn.append(f"{s1_without_gt:,} S1 rows have no ground-truth line at all (treated as singletons)")
    miss_t = int((truth_rec < 0).sum())
    if miss_t:
        warn.append(f"{miss_t:,} true ids not found in the train TSVs")
    if rows_dup:
        warn.append(f"{rows_dup:,} duplicate candidates inside rows")
    train_topk = int(row_lens.max()) if len(row_lens) else 0
    out(f"candidates per S1 row: max {train_topk}, mean {row_lens.mean():.1f} -- predict_matches.py trims "
        f"test rows to the same {train_topk} (rank features/g_n would otherwise be out of range)")
    out(f"S1 rows with zero candidates: {(row_lens == 0).mean() * 100:.2f}% | true pairs in candidates: "
        f"{tc.sum() / max(nt.sum(), 1) * 100:.2f}% (blocking recall at topk used)")
    const = [FEATURES[i] for i in feat_idx if X[:, i].min() == X[:, i].max()]
    if const:
        warn.append(f"constant (useless) features: {const}")

    if a.val_country:
        out(f"LEAVE-ONE-COUNTRY-OUT: validating on {a.val_country}, training on {sorted(set(country[~val_rows]))}")
    is_val = val_rows[row]
    va_idx = np.flatnonzero(is_val)
    tr_all = np.flatnonzero(~is_val)
    tr_idx = tr_all
    out(f"train pairs {len(tr_idx):,} (S1 rows {(~val_rows).sum():,}"
        + (f"; negatives: all ranked <{a.neg_hard_k} + {a.neg_frac:.0%} of the rest, weighted" if a.neg_frac < 1 else "")
        + f") | val pairs {len(va_idx):,} | val S1 rows {val_rows.sum():,}")

    vrows = np.flatnonzero(val_rows)
    remap = -np.ones(nr, np.int64)
    remap[vrows] = np.arange(len(vrows))
    vr = remap[row[va_idx]]
    vy, vkey, vnt, vtc = y[va_idx], b_idx[va_idx], nt[vrows], tc[vrows]
    nv = len(vrows)
    X1 = lambda idx: X[np.ix_(idx, feat_idx)] if len(feat_idx) < X.shape[1] else X[idx]

    # ------------------------------------------------------------------ stage 1
    t = time.time()
    bst = train_lgb(X1(tr_idx), y[tr_idx], X1(va_idx), y[va_idx], names1, a, wtr=wt[tr_idx])
    best_it = bst.best_iteration or bst.current_iteration()
    out(f"stage 1 LightGBM: best_iteration={best_it} in {time.time() - t:.0f}s")
    p_va1 = ef.floor_p(bst.predict(X1(va_idx), num_iteration=best_it, num_threads=a.workers), a.tau0)
    sub = tr_idx if len(tr_idx) <= 2_000_000 else rng.choice(tr_idx, 2_000_000, replace=False)
    auc_tr = auc(y[sub], bst.predict(X1(sub), num_iteration=best_it, num_threads=a.workers))
    auc_va = auc(vy, p_va1)
    extras = []
    if a.n_models > 1:
        t = time.time()
        raw = [bst.predict(X1(va_idx), num_iteration=best_it, num_threads=a.workers)]
        cand_ex = []
        for k in range(1, a.n_models):
            bk = train_lgb(X1(tr_idx), y[tr_idx], X1(va_idx), y[va_idx], names1, a, wtr=wt[tr_idx],
                           seed=a.seed + 100 * k)
            itk = bk.best_iteration or bk.current_iteration()
            raw.append(bk.predict(X1(va_idx), num_iteration=itk, num_threads=a.workers))
            cand_ex.append((bk, itk))
        p_ens_va1 = ef.floor_p(np.mean(raw, axis=0), a.tau0)
        out(f"stage 1 seed ensemble: trained {a.n_models - 1} extra model(s) in {time.time() - t:.0f}s "
            f"(judged on the FINAL pipeline below, after stage 2)")
    else:
        p_ens_va1, cand_ex = None, []
    t = time.time()
    rule1, grid1 = grid_search(vr, p_va1, vy, vkey, vnt, nv, rules)
    F1 = score_pred(vr, ef.apply_rule(rule1, vr, p_va1, vkey, nv), vy, vnt, nv)[0]
    out(f"stage 1: macro F0.5 {F1.mean():.4f} with {rule_str(rule1)}  (rule grid {len(grid1)} in {time.time() - t:.0f}s)")
    if "thresh" in rules and "ef" in rules:
        bt = max(v for k, v in grid1.items() if k[0] == "thresh")
        be = max(v for k, v in grid1.items() if k[0] == "ef")
        out(f"   best thresholds rule {bt:.4f} | best expected-F0.5 rule {be:.4f}")

    # ------------------------------------------------------------------ stage 2
    final_p, final_rule, bst2, best_it2 = p_va1, rule1, None, 0
    s2_info = {}
    if a.stage2:
        out("\n==================== STAGE 2 (consensus) ====================")
        t = time.time()
        p1_all = np.zeros(npairs)
        p1_all[va_idx] = p_va1
        rows_tr = np.flatnonzero(~val_rows)
        fold_of_row = -np.ones(nr, np.int64)
        fold_of_row[rows_tr] = rng.randint(a.oof_folds, size=len(rows_tr))
        f_tr_idx, f_all = fold_of_row[row[tr_idx]], fold_of_row[row[tr_all]]
        for k in range(a.oof_folds):
            fit = tr_idx[f_tr_idx != k]
            tgt = tr_all[f_all == k]
            b = train_lgb(X1(fit), y[fit], None, None, names1, a, rounds=best_it, early=False,
                          seed=a.seed + 10 + k, wtr=wt[fit])
            p1_all[tgt] = ef.floor_p(b.predict(X1(tgt), num_threads=a.workers), a.tau0)
            ef.log(f"  OOF fold {k + 1}/{a.oof_folds} done")
        oof_auc = auc(y[tr_all], p1_all[tr_all]) if len(tr_all) <= 5_000_000 else float("nan")
        out(f"out-of-fold stage-1 p for train rows in {time.time() - t:.0f}s (OOF AUC {oof_auc:.5f} vs val {auc_va:.5f})")
        t = time.time()
        sel, F2 = ef.consensus_features(row, p1_all, b_raw, is_s3, rec, nr, a.workers, a.tau0)
        sel_val = is_val[sel]
        s_tr, s_va = sel[~sel_val], sel[sel_val]
        lost = int(vy[p_va1 < a.tau0].sum())
        out(f"stage-2 pairs: {len(sel):,} of {npairs:,} ({len(sel) / npairs * 100:.1f}%) have p1 >= {a.tau0} | "
            f"val positives below tau0 (unreachable for stage 2): {lost:,} of {int(vy.sum()):,} | {time.time() - t:.0f}s")
        Xs = ef.stage2_matrix(X, sel, F2, feat_idx)
        t = time.time()
        bst2 = train_lgb(Xs[~sel_val], y[s_tr], Xs[sel_val], y[s_va], names2, a, wtr=wt[s_tr])
        best_it2 = bst2.best_iteration or bst2.current_iteration()
        p2 = ef.floor_p(bst2.predict(Xs[sel_val], num_iteration=best_it2, num_threads=a.workers), a.tau0)
        out(f"stage 2 LightGBM: best_iteration={best_it2} in {time.time() - t:.0f}s")
        pos_in_va = np.searchsorted(va_idx, s_va)
        p_va2 = p_va1.copy()
        p_va2[pos_in_va] = p2
        rule2, grid2 = grid_search(vr, p_va2, vy, vkey, vnt, nv, rules)
        F2m = score_pred(vr, ef.apply_rule(rule2, vr, p_va2, vkey, nv), vy, vnt, nv)[0]
        out(f"stage 2: macro F0.5 {F2m.mean():.4f} with {rule_str(rule2)}   (stage 1: {F1.mean():.4f}, "
            f"delta {(F2m.mean() - F1.mean()) * 100:+.3f} pts)")
        gain2 = dict(zip(names2, bst2.feature_importance("gain", iteration=best_it2)))
        tg2 = sum(gain2.values()) or 1
        out("stage 2 gain share: consensus features " + f"{sum(gain2[f] for f in CONS_FEATS) / tg2 * 100:.1f}% | top: " +
            ", ".join(f"{f}={v / tg2 * 100:.1f}%" for f, v in sorted(gain2.items(), key=lambda kv: -kv[1])[:12]))
        s2_info = {"stage1_val_macro": F1.mean(), "stage2_val_macro": F2m.mean()}
        if F2m.mean() > F1.mean():
            final_p, final_rule = p_va2, rule2
            out("=> stage 2 is used (saved as model2.txt)")
        else:
            bst2 = None
            out("=> stage 2 did not help on validation; stage 1 alone is used")
        del Xs

    # ------------------------------------------------------------------ seed ensemble, judged end to end
    if p_ens_va1 is not None:
        pe = p_ens_va1.copy()
        if bst2 is not None:
            sv, F2v = ef.consensus_features(vr, p_ens_va1, b_raw[va_idx], is_s3[va_idx], rec, nv, a.workers, a.tau0)
            if len(sv):
                pe[sv] = ef.floor_p(bst2.predict(ef.stage2_matrix(X[va_idx], sv, F2v, feat_idx),
                                                 num_iteration=best_it2, num_threads=a.workers), a.tau0)
        rule_e, grid_e = grid_search(vr, pe, vy, vkey, vnt, nv, rules)
        m_e = grid_e[max(grid_e, key=grid_e.get)]
        m_s = score_pred(vr, ef.apply_rule(final_rule, vr, final_p, vkey, nv), vy, vnt, nv)[0].mean()
        use = m_e > m_s
        out(f"seed ensemble of {a.n_models} (final pipeline): macro F0.5 {m_e:.4f} vs single model {m_s:.4f} -> "
            f"{'ensemble USED' if use else 'single model kept'}")
        if use:
            final_p, final_rule, extras, p_va1 = pe, rule_e, cand_ex, p_ens_va1
            auc_va = auc(vy, p_va1)

    # ------------------------------------------------------------------ prefilter cascade (speed only)
    pre_info = None
    if a.prefilter and not extras:
        t = time.time()
        pre = train_lgb(X1(tr_idx), y[tr_idx], None, None, names1, a, rounds=80, early=False, wtr=wt[tr_idx],
                        over={"learning_rate": 0.2, "num_leaves": 31, "feature_fraction": 1.0})
        q_va = pre.predict(X1(va_idx), num_threads=a.workers)
        matters = p_va1 >= a.tau0 / 2
        cut = 0.5 * float(q_va[matters].min()) if matters.any() else 0.0
        skip = q_va < cut
        # replay the exact predict path on validation with the cascade on
        p1c = np.where(skip, 0.0, p_va1)
        pc_final = p1c.copy()
        if bst2 is not None:
            sv, F2v = ef.consensus_features(vr, p1c, b_raw[va_idx], is_s3[va_idx], rec, nv, a.workers, a.tau0)
            if len(sv):
                pc_final[sv] = ef.floor_p(bst2.predict(ef.stage2_matrix(X[va_idx], sv, F2v, feat_idx),
                                                       num_iteration=best_it2, num_threads=a.workers), a.tau0)
        mc = score_pred(vr, ef.apply_rule(final_rule, vr, pc_final, vkey, nv), vy, vnt, nv)[0].mean()
        mf = score_pred(vr, ef.apply_rule(final_rule, vr, final_p, vkey, nv), vy, vnt, nv)[0].mean()
        use = skip.mean() >= 0.3 and mc >= mf - 1e-12
        out(f"\nprefilter: skips the big model on {skip.mean() * 100:.1f}% of val pairs (cut q<{cut:.2e}; every pair "
            f"with p1>={a.tau0 / 2} kept) | val F0.5 with cascade {mc:.5f} vs without {mf:.5f} -> "
            f"{'USED at predict (training countries only)' if use else 'not used'} ({time.time() - t:.0f}s)")
        if use:
            pre.save_model(os.path.join(a.out_dir, "model_pre.txt"))
            pre_info = {"cut": cut, "val_drop_frac": float(skip.mean())}

    # ------------------------------------------------------------------ analysis of the final scores
    p_va = final_p
    pred = ef.apply_rule(final_rule, vr, p_va, vkey, nv)
    F, tp, npred = score_pred(vr, pred, vy, vnt, nv)
    macro = F.mean()
    _, grid = grid_search(vr, p_va, vy, vkey, vnt, nv, ("thresh",))
    if final_rule["kind"] == "thresh":
        tt, tx = final_rule["t_top"], final_rule["t_extra"]
        if tt in (THRESH[0], THRESH[-1]) or tx in (THRESH[0], THRESH[-1]):
            warn.append(f"best threshold on the grid edge (t_top={tt}, t_extra={tx}) -- widen THRESH")
    elif final_rule["empty_bias"] in (EF_B[0], EF_B[-1]) or final_rule["m_extra"] == EF_M[-1]:
        warn.append(f"expected-F0.5 rule on its grid edge ({final_rule}) -- widen EF_B / EF_M")
    if auc_tr - auc_va > 0.005:
        warn.append(f"train AUC {auc_tr:.4f} vs val {auc_va:.4f}: overfitting -- raise min_data_in_leaf / lower leaves")
    for w in warn:
        out(f"  !! {w}")

    out("\n==================== HEADLINE (validation S1 rows, singletons included) ====================")
    ceiling = row_f_half(vtc, vtc, vnt).mean()
    single_best = max(v for (kind, a1, b1, o1), v in grid.items() if a1 == b1 and not o1)
    out(f"macro F0.5 = {macro:.4f}   ({'stage 2' if bst2 is not None else 'stage 1'}; {rule_str(final_rule)})")
    out(f"  single global threshold, no owner-exclusive (old approach): {single_best:.4f}")
    out(f"  blocking ceiling (perfect matcher on these candidates):     {ceiling:.4f}")
    out(f"  gap to ceiling: {ceiling - macro:.4f}  | stage-1 val AUC {auc_va:.5f} (train {auc_tr:.5f})")

    out("\n==================== LOSS LEDGER: where 1 - F0.5 goes ====================")
    loss = {}
    L = lambda m, v=1.0: np.where(m, v, 0.0)
    has_t, has_c = vnt > 0, vtc > 0
    loss["singleton_false_merge"] = L((vnt == 0) & (npred > 0))
    loss["blocking_total_miss (abstained, nothing findable)"] = L(has_t & ~has_c & (npred == 0))
    loss["blocking_total_miss + wrong pick"] = L(has_t & ~has_c & (npred > 0))
    loss["abstained_findable"] = L(has_t & has_c & (npred == 0))
    loss["wrong_pick_findable"] = L(has_t & has_c & (npred > 0) & (tp == 0))
    part = tp > 0
    F_nofp = row_f_half(tp, tp, vnt)
    F_cand = row_f_half(vtc, vtc, vnt)
    loss["partial_extra_fp"] = np.where(part, F_nofp - F, 0)
    loss["partial_recall_matcher (in candidates, not predicted)"] = np.where(part, F_cand - F_nofp, 0)
    loss["partial_recall_blocking (not in candidates)"] = np.where(part, 1 - F_cand, 0)
    tot = sum(v.sum() for v in loss.values()) / nv
    fix = {"singleton_false_merge": "raise t_top / better singleton features",
           "blocking_total_miss (abstained, nothing findable)": "blocking: keys/expansion/topk",
           "blocking_total_miss + wrong pick": "blocking + raise t_top",
           "abstained_findable": "lower t_top / model recall",
           "wrong_pick_findable": "model features (discriminate near-duplicates)",
           "partial_extra_fp": "raise t_extra / owner-exclusive / stage 2",
           "partial_recall_matcher (in candidates, not predicted)": "lower t_extra / model / stage 2",
           "partial_recall_blocking (not in candidates)": "blocking: topk / sibling expansion"}
    out(f"{'category':58s} {'rows':>8} {'F0.5 pts lost':>14}  lever")
    for k, v in sorted(loss.items(), key=lambda kv: -kv[1].sum()):
        out(f"{k:58s} {int((v > 0).sum()):8,} {v.sum() / nv * 100:13.3f}  {fix[k]}")
    out(f"{'TOTAL (must equal 1 - F0.5)':58s} {'':8s} {tot * 100:13.3f}  check: {(1 - macro) * 100:.3f}")
    assert abs(tot - (1 - macro)) < 1e-6, "ledger does not add up"
    blk = sum(loss[k].sum() for k in loss if "blocking" in k) / nv
    out(f"=> blocking-caused {blk * 100:.2f} pts | matcher/decision-caused {(tot - blk) * 100:.2f} pts")
    # rejected positives by probability band: which can a rule fix, which need features
    rej = vy & ~pred
    bands = [(0, 0.01), (0.01, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
    out("rejected TRUE pairs by p (low p = model/features, mid p = decision rule): " + "  ".join(
        f"[{lo},{min(hi, 1)}):{int((rej & (p_va >= lo) & (p_va < hi)).sum()):,}" for lo, hi in bands))
    fa = pred & ~vy
    out("accepted FALSE pairs by p: " + "  ".join(
        f"[{lo},{min(hi, 1)}):{int((fa & (p_va >= lo) & (p_va < hi)).sum()):,}" for lo, hi in bands))
    # who do the false positives belong to? (full ground truth knows every record's owner)
    fa_i = np.flatnonzero(fa & (b_raw[va_idx] >= 0))
    if len(fa_i):
        own = pc.is_in(rec["ids"].take(pa.array(b_raw[va_idx][fa_i])), value_set=owned).to_numpy(zero_copy_only=False)
        sing = vnt[vr[fa_i]] == 0
        out(f"accepted FALSE pairs: {own.mean() * 100:.1f}% are records that truly belong to ANOTHER S1 "
            f"(competition: reverse/owner features help), {100 - own.mean() * 100:.1f}% belong to no S1 (unowned records)"
            f" | in singleton rows: {sing.sum():,} FPs, {own[sing].mean() * 100 if sing.any() else 0:.1f}% owned by another S1")

    out("\n==================== SEGMENTS ====================")
    lnb = cand["n_before_prune"].to_numpy()[vrows]
    s1_addr_empty = rec["addr_empty"][np.maximum(s1_rec[vrows], 0)]
    s1_native = rec["native"][np.maximum(s1_rec[vrows], 0)]
    segs = {
        "country": country[vrows],
        "#true": np.select([vnt == 0, vnt == 1, vnt == 2, vnt == 3, vnt <= 5], ["0", "1", "2", "3", "4-5"], "6+"),
        "union size": np.select([lnb == 0, lnb < 10, lnb < 100, lnb < 1000], ["0", "1-9", "10-99", "100-999"], "1000+"),
        "S1 addr empty": s1_addr_empty.astype(str),
        "S1 native script": s1_native.astype(str),
    }
    for name, lab in segs.items():
        out(f"-- by {name}")
        out(f"   {'value':10s} {'rows':>8} {'F0.5':>7} {'blocking':>9} {'fp-side':>8} {'recall-side':>11}")
        for v in sorted(set(lab)):
            m = lab == v
            b = sum(loss[k][m].sum() for k in loss if "blocking" in k) / m.sum()
            fpl = (loss["singleton_false_merge"][m].sum() + loss["partial_extra_fp"][m].sum()
                   + loss["wrong_pick_findable"][m].sum()) / m.sum()
            rcl = (loss["abstained_findable"][m].sum()
                   + loss["partial_recall_matcher (in candidates, not predicted)"][m].sum()) / m.sum()
            out(f"   {v:10s} {m.sum():8,} {F[m].mean():7.4f} {b * 100:8.2f}p {fpl * 100:7.2f}p {rcl * 100:10.2f}p")

    out("\n==================== THRESHOLD LANDSCAPE (final scores, threshold rule) ====================")
    (_, bt_t, bt_x, bt_o) = max(grid, key=grid.get)
    out("t_extra sweep at best t_top/owner-exclusive:  " +
        "  ".join(f"{x}:{grid[('thresh', bt_t, x, bt_o)]:.4f}" for x in THRESH))
    out("t_top sweep at best t_extra/owner-exclusive:  " +
        "  ".join(f"{x}:{grid[('thresh', x, bt_x, bt_o)]:.4f}" for x in THRESH))
    out(f"owner-exclusive on vs off: {grid[('thresh', bt_t, bt_x, True)]:.4f} vs {grid[('thresh', bt_t, bt_x, False)]:.4f}"
        "  (sampled validation hides competing S1 rows -- judge owner-exclusive on the test contention report)")

    out("\n==================== MODEL (stage 1) ====================")
    bins = np.clip((p_va1 * 10).astype(int), 0, 9)
    out("calibration (val): " + "  ".join(
        f"[{b / 10:.1f},{(b + 1) / 10:.1f}) n={int((bins == b).sum()):,} pos={vy[bins == b].mean() * 100:.1f}%"
        for b in range(10) if (bins == b).any()))
    gain = dict(zip(names1, bst.feature_importance("gain", iteration=best_it)))
    tg = sum(gain.values()) or 1
    out("gain share by group: " + ", ".join(
        f"{g}={sum(gain.get(f, 0) for f in fs) / tg * 100:.1f}%" for g, fs in FEATURE_GROUPS.items()))
    out("top 25 features: " + ", ".join(f"{f}={v / tg * 100:.1f}%" for f, v in
                                         sorted(gain.items(), key=lambda kv: -kv[1])[:25]))

    if a.ablation:
        out("\n==================== ABLATION (stage 1, retrain without each group, reduced data) ====================")
        sub_tr = rng.choice(tr_idx, int(len(tr_idx) * a.ablation_frac), replace=False)
        rounds = max(100, min(best_it, 600))

        def score(cols, seed=None):
            Xa = lambda idx: X[np.ix_(idx, cols)]
            b2 = train_lgb(Xa(sub_tr), y[sub_tr], None, None, [FEATURES[i] for i in cols], a,
                           rounds=rounds, early=False, seed=seed, wtr=wt[sub_tr])
            pp = b2.predict(Xa(va_idx), num_threads=a.workers)
            return grid_search(vr, pp, vy, vkey, vnt, nv, rules,
                               grid=[0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999])
        base_r, g0 = score(feat_idx)
        base = max(g0.values())
        base2 = max(score(feat_idx, a.seed + 1)[1].values())
        out(f"reference (used features, {a.ablation_frac:.0%} of train, {rounds} rounds): {base:.4f} | "
            f"reseeded: {base2:.4f} -> differences under ~{abs(base - base2) * 100 * 2:.2f} pts are noise")
        for g, fs in FEATURE_GROUPS.items():
            if g in drop:
                continue
            keep = np.array([i for i in feat_idx if FEATURES[i] not in fs])
            s = max(score(keep)[1].values())
            out(f"  without {g:10s}: {s:.4f}  (group worth {(base - s) * 100:+.3f} pts)")

    # ------------------------------------------------------------------ examples
    ids_all = rec["ids"]
    nos, adr = rec["nosuf"], rec["addr"]
    txt = lambda i: (f"{ids_all[i].as_py()} | {nos[i].as_py()} | {adr[i].as_py()}" if i >= 0 else "(missing)")
    lines = []
    for cat in ["singleton_false_merge", "wrong_pick_findable", "abstained_findable",
                "blocking_total_miss (abstained, nothing findable)", "partial_extra_fp",
                "partial_recall_matcher (in candidates, not predicted)"]:
        rows_c = np.flatnonzero(loss[cat] > 0)
        if not len(rows_c):
            continue
        lines.append(f"\n########## {cat}: {len(rows_c):,} rows (showing {min(a.examples, len(rows_c))}) ##########")
        for r in rng.choice(rows_c, min(a.examples, len(rows_c)), replace=False):
            gr = vrows[r]
            lines.append(f"\nS1 {txt(s1_rec[gr])}   [{country[gr]}, #true={int(vnt[r])}, in-cands={int(vtc[r])}, union={lnb[r]}]")
            m = np.flatnonzero(vr == r)
            o = m[np.argsort(-p_va[m])][:8]
            for j in o:
                tag = ("PRED " if pred[j] else "     ") + ("TRUE" if vy[j] else "    ")
                lines.append(f"   {tag} p={p_va[j]:.3f} p1={p_va1[j]:.3f} rank={int(rank[va_idx[j]])}  {txt(b_raw[va_idx[j]])}")
            for tr in truth_rec[truth_row == gr]:
                if not ((vr == r) & (vkey == tr) & vy).any():
                    lines.append(f"   MISSING-FROM-CANDIDATES  {txt(tr)}")
    with open(os.path.join(a.out_dir, "loss_examples.txt"), "w") as f:
        f.write("\n".join(lines))

    # ------------------------------------------------------------------ save
    bst.save_model(os.path.join(a.out_dir, "model.txt"), num_iteration=best_it)
    extra_meta = []
    for k, (bk, itk) in enumerate(extras, 1):
        fn = f"model_s1_{k}.txt"
        bk.save_model(os.path.join(a.out_dir, fn), num_iteration=itk)
        extra_meta.append(fn)
    if bst2 is not None:
        bst2.save_model(os.path.join(a.out_dir, "model2.txt"), num_iteration=best_it2)
    thr = final_rule if final_rule["kind"] == "thresh" else {"t_top": 0.7, "t_extra": 0.7}
    meta = {"features": FEATURES, "feat_idx": feat_idx.tolist(), "dropped_groups": drop,
            "stage2": bst2 is not None, "tau0": a.tau0, "cons_feats": CONS_FEATS,
            "rule": final_rule, "prefilter": pre_info, "p_floor": True, "one_owner": multi_owner == 0,
            "stage1_extra": extra_meta,
            # v2-compatible keys
            "t_top": thr["t_top"], "t_extra": thr["t_extra"], "owner_exclusive": final_rule["owner_exclusive"],
            "topk_use": a.topk_use, "train_topk": train_topk, "best_iteration": best_it, "best_iteration2": best_it2,
            "val_macro_f05": macro, "val_ceiling": ceiling, "val_auc": auc_va,
            "val_pred_empty_rate": float((npred == 0).mean()), "val_avg_pred_per_row": float(npred.mean()),
            "val_country": a.val_country, "train_countries": sorted(set(country[~val_rows])), **s2_info}
    with open(os.path.join(a.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    out(f"\nsaved model/meta/report/examples to {a.out_dir}/  (total {time.time() - T0:.0f}s)")
    with open(os.path.join(a.out_dir, "report.txt"), "w") as f:
        f.write("\n".join(_REPORT))


if __name__ == "__main__":
    main()

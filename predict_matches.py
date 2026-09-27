"""
Test-time inference: v7 test candidates -> output/matching_results.tsv + candidate_pairs.tsv.

  python predict_matches.py --candidates cand_test.parquet --data-dir dataset \
      --translit-dict translit_dict.json --model-dir matcher_v3 --out-dir output --workers 12

Streams the candidates parquet in --batch-s1 row batches (features, stage-1, optional
stage-2, per-row decision), writes candidate_pairs.tsv per batch, keeps only accepted
pairs, then applies owner-exclusive globally and writes matching_results.tsv.
Works with v2 model dirs (thresholds) and v3 model dirs (stage 2, expected-F0.5 rule).

--unseen-model-dir DIR: rows whose country is not among the main model's training
countries (France) are scored and decided by the model in DIR instead (e.g. one
trained with --drop-groups, chosen with a --val-country experiment).

DIAGNOSTICS (no labels needed), written to the log and --out-dir/diagnostics.txt:
  * per country: accepted pairs/row BEFORE owner-exclusive, share of accepted pairs
    whose S2/S3 record is claimed by 2+ S1 rows (every record has one true owner, so
    the losing claims are certain false positives), empty rate, confident-top share
  * feature drift: per-country mean of every feature on each row's rank-0 candidate,
    in standard deviations from the training countries -- shows WHICH inputs look
    different for an unseen country
  * contested_examples.txt: records claimed by several S1 rows, with names/addresses
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


def iter_batches(paths, batch_rows, columns=None):
    for p in paths:
        for b in pq.ParquetFile(p).iter_batches(batch_size=batch_rows, columns=columns):
            yield pa.Table.from_batches([b]).combine_chunks()


def load_bundle(model_dir):
    import lightgbm as lgb
    meta_path = os.path.join(model_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise SystemExit(f"{meta_path} not found -- run train_matcher.py first (or point --model-dir at it)")
    meta = json.load(open(meta_path))
    if meta["features"] != ef.FEATURES:
        raise SystemExit(f"{model_dir}: feature list differs from training -- er_features.py changed since training")
    b = {"dir": model_dir, "meta": meta,
         "bst": lgb.Booster(model_file=os.path.join(model_dir, "model.txt")),
         "feat_idx": np.array(meta.get("feat_idx", list(range(len(ef.FEATURES))))),
         "rule": meta.get("rule") or {"kind": "thresh", "t_top": meta["t_top"], "t_extra": meta["t_extra"],
                                      "owner_exclusive": meta["owner_exclusive"]},
         "bst2": None, "pre": None,
         "extra": [lgb.Booster(model_file=os.path.join(model_dir, f)) for f in meta.get("stage1_extra", [])]}
    if meta.get("prefilter"):
        b["pre"] = lgb.Booster(model_file=os.path.join(model_dir, "model_pre.txt"))
    if meta.get("stage2"):
        if meta.get("cons_feats") != ef.CONS_FEATS:
            raise SystemExit(f"{model_dir}: stage-2 feature list differs -- er_features.py changed since training")
        b["bst2"] = lgb.Booster(model_file=os.path.join(model_dir, "model2.txt"))
    return b


def score(bundle, X, P, b_idx, is_s3, rec, nb, workers, pre_ok=None):
    """pre_ok: per-pair bool, where the prefilter cascade may be used (training-country rows)."""
    fx = bundle["feat_idx"]
    X1 = X if len(fx) == X.shape[1] else X[:, fx]
    if not len(X1):
        return np.zeros(0)
    if bundle["extra"]:
        p = np.mean([bundle["bst"].predict(X1, num_threads=workers)] +
                    [e.predict(X1, num_threads=workers) for e in bundle["extra"]], axis=0)
    elif bundle["pre"] is not None and pre_ok is not None and pre_ok.any():
        q = bundle["pre"].predict(X1, num_threads=workers)
        full = ~pre_ok | (q >= bundle["meta"]["prefilter"]["cut"])
        p = np.zeros(len(q))
        p[full] = bundle["bst"].predict(X1[full], num_threads=workers)
        bundle["n_skipped"] = bundle.get("n_skipped", 0) + int((~full).sum())
    else:
        p = bundle["bst"].predict(X1, num_threads=workers)
    tau0 = bundle["meta"].get("tau0")
    floored = bundle["meta"].get("p_floor", False)
    if floored:
        p = ef.floor_p(p, tau0)
    if bundle["bst2"] is not None and len(p):
        sel, F2 = ef.consensus_features(P["row"], p, b_idx, is_s3, rec, nb, workers, tau0)
        if len(sel):
            p = p.copy()
            p2 = bundle["bst2"].predict(ef.stage2_matrix(X1, sel, F2), num_threads=workers)
            p[sel] = ef.floor_p(p2, tau0) if floored else p2
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="v7 test parquet(s), comma-separated")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--translit-dict", default="translit_dict.json")
    ap.add_argument("--model-dir", default="matcher_v3")
    ap.add_argument("--unseen-model-dir", default="", help="model for countries the main model never trained on")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--batch-s1", type=int, default=30_000, help="S1 rows per batch (memory ~ batch*topk*97*4 bytes)")
    ap.add_argument("--topk-use", type=int, default=0,
                    help="candidates per row to score: 0 = same as training (default), -1 = all")
    ap.add_argument("--contested-examples", type=int, default=30, help="per country")
    ap.add_argument("--no-prefilter", action="store_true", help="always run the full model on every pair")
    ap.add_argument("--save-scores", type=int, default=1,
                    help="1 = also write out_dir/scores.parquet (every pair with p > 0) so redecide.py can re-apply "
                         "a different decision rule in minutes without re-scoring")
    ap.add_argument("--single-model", action="store_true",
                    help="ignore a saved seed ensemble and use only the first stage-1 model (faster)")
    ap.add_argument("--owner-exclusive", default="auto", choices=["auto", "on", "off"],
                    help="auto = ON when training verified every S2/S3 id has one owner (validation cannot "
                         "judge it: its sample hides the competing S1 rows), else the model's choice")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    T0 = time.time()
    main_b = load_bundle(a.model_dir)
    if a.single_model and main_b["extra"]:
        ef.log(f"--single-model: ignoring {len(main_b['extra'])} extra stage-1 model(s)")
        main_b["extra"] = []
    meta = main_b["meta"]
    unseen_b = load_bundle(a.unseen_model_dir) if a.unseen_model_dir else None
    train_countries = set(meta.get("train_countries") or [])
    ef.log(f"model {a.model_dir}: stage1 models={1 + len(main_b['extra'])} stage2={main_b['bst2'] is not None} "
           f"prefilter={main_b['pre'] is not None and not a.no_prefilter} "
           f"rule={main_b['rule']}")
    if unseen_b:
        if not train_countries:
            raise SystemExit("--unseen-model-dir needs a v3 main model (meta.json has train_countries)")
        ef.log(f"unseen-country model {a.unseen_model_dir}: stage2={unseen_b['bst2'] is not None} "
               f"rule={unseen_b['rule']} -- used for countries not in {sorted(train_countries)}")
    oe_global = {"on": True, "off": False}.get(a.owner_exclusive,
                                               True if meta.get("one_owner") else main_b["rule"]["owner_exclusive"])
    ef.log(f"owner-exclusive: {'ON' if oe_global else 'off'} ({a.owner_exclusive})")
    train_topk = meta.get("topk_use") or meta.get("train_topk") or None
    if unseen_b is not None:
        ut = unseen_b["meta"].get("topk_use") or unseen_b["meta"].get("train_topk")
        if ut and train_topk and ut != train_topk:
            raise SystemExit(f"main model trained on {train_topk} candidates/row, unseen model on {ut}; retrain one")
    tk = None if a.topk_use < 0 else (a.topk_use or train_topk)
    ef.log(f"scoring the first {tk or 'all'} candidates of each row"
           + (f" (model trained on {train_topk}/row)" if train_topk else ""))
    paths = a.candidates.split(",")
    with open(a.translit_dict, encoding="utf-8") as f:
        translit = json.load(f)

    s1_tbl = ef.read_tsv_arrow(os.path.join(a.data_dir, a.split, f"{a.split}_source1.tsv"))
    all_ids = s1_tbl["entity_id"].combine_chunks()
    s1_country = dict(zip(all_ids.to_pylist(), s1_tbl["country"].to_pylist()))
    del s1_tbl

    # ---------------- pass 1: which records are needed (cheap, streamed)
    t = time.time()
    s1_parts, uniq, test_k = [], [], 0
    for c in iter_batches(paths, 200_000, ["source1_entity_id", "candidate_entity_ids"]):
        s1_parts.append(c["source1_entity_id"].combine_chunks())
        lists = c["candidate_entity_ids"].combine_chunks()
        test_k = max(test_k, int(pc.max(pc.list_value_length(lists)).as_py() or 0))
        if tk:
            lists = pc.list_slice(lists, 0, tk)
        uniq.append(pc.unique(pc.list_flatten(lists)))
    s1_seen = pa.concat_arrays(s1_parts)
    cand_ids = pc.unique(pa.concat_arrays(uniq)) if uniq else pa.array([], pa.string())
    del uniq
    n_s1 = len(s1_seen)
    if train_topk and test_k > train_topk and tk:
        ef.log(f"  note: test rows have up to {test_k} candidates, trimmed to {tk} to match training. To use all "
               f"{test_k}, rebuild TRAIN candidates with blocking --topk {test_k} and retrain.")
    if train_topk and test_k < train_topk:
        ef.log(f"  !! test rows have only {test_k} candidates but the model was trained on {train_topk}: "
               f"rank/row-size features are shifted -- rebuild TEST candidates with --topk {train_topk}")
    dup = n_s1 - len(pc.unique(s1_seen))
    if dup:
        raise SystemExit(f"{dup} duplicate S1 rows across candidate parquets")
    missing = len(all_ids) - int(pc.sum(pc.is_in(all_ids, value_set=s1_seen)).as_py() or 0)
    ef.log(f"test S1: {len(all_ids):,} | in candidates: {n_s1:,} | unique candidate records: "
           f"{len(cand_ids):,} ({time.time() - t:.0f}s)")
    if missing:
        ef.log(f"  !! {missing:,} test S1 rows have NO candidate row (a country parquet missing?) -- written empty")
    t = time.time()
    rec = ef.normalize_records(ef.read_records(a.data_dir, a.split, pa.concat_arrays([s1_seen, cand_ids])),
                               translit, a.workers)
    rec = ef.attach_name_freq(rec, a.data_dir, a.split, translit, a.workers)
    del cand_ids
    ef.log(f"normalized {len(rec['ids']):,} records in {time.time() - t:.0f}s")

    s1_list = s1_seen.to_pylist()
    ctry = np.array([s1_country.get(s, "?") for s in s1_list])
    countries = sorted(set(ctry))
    cix = {c: i for i, c in enumerate(countries)}
    ctry_i = np.array([cix[c] for c in ctry], np.int64)
    nF, nC = len(ef.FEATURES), len(countries)
    dsum = np.zeros((nC, nF))
    dsq = np.zeros((nC, nF))
    dn = np.zeros(nC)
    row_top_p = np.zeros(n_s1, np.float32)
    s1_rec_all = np.zeros(n_s1, np.int64)

    # ---------------- pass 2: score batches, stream candidate_pairs.tsv
    os.makedirs(a.out_dir, exist_ok=True)
    fc = open(os.path.join(a.out_dir, "candidate_pairs.tsv"), "w")
    fc.write("source1_entity_id\tcandidate_entity_ids\n")
    acc_row, acc_key, acc_p = [], [], []
    sw = None
    if a.save_scores:
        sw = pq.ParquetWriter(os.path.join(a.out_dir, "scores.parquet"),
                              pa.schema([("row", pa.int32()), ("cand", pa.string()), ("p", pa.float32())]))
    errors, base, neg_ctr = [], 0, 0
    R = len(rec["ids"]) + 1
    is_s3_col = ef.FEATURES.index("is_s3")
    for c in iter_batches(paths, a.batch_s1):
        tb = time.time()
        nb = c.num_rows
        P = ef.explode_candidates(c, tk)
        s1_rec = ef.rec_index(rec, c["source1_entity_id"].combine_chunks())
        s1_rec_all[base:base + nb] = s1_rec
        X, b_idx = ef.build_features(c, P, rec, s1_rec, a.workers)
        t_feat = time.time() - tb
        is_s3 = X[:, is_s3_col] > 0.5
        miss = b_idx < 0
        key = b_idx.copy()
        key[miss] = -2 - neg_ctr - np.arange(miss.sum())
        neg_ctr += int(miss.sum())

        bc = ctry_i[base:base + nb]
        r0 = P["rank"] == 0
        c0 = bc[P["row"][r0]]
        X0 = X[r0].astype(np.float64)
        for ci_ in np.unique(c0):
            mk = c0 == ci_
            dsum[ci_] += X0[mk].sum(0)
            dsq[ci_] += (X0[mk] ** 2).sum(0)
        dn += np.bincount(c0, minlength=nC)
        del X0

        tm = time.time()
        pre_ok = None
        if not a.no_prefilter and train_countries:
            pre_ok = np.isin(ctry[base:base + nb], list(train_countries))[P["row"]]
        p = score(main_b, X, P, b_idx, is_s3, rec, nb, a.workers, pre_ok)
        pred = ef.apply_rule(main_b["rule"], P["row"], p, key, nb, oe_local=False)
        if unseen_b is not None:
            row_unseen = ~np.isin(ctry[base:base + nb], list(train_countries))
            if row_unseen.any():
                pu = score(unseen_b, X, P, b_idx, is_s3, rec, nb, a.workers)
                predu = ef.apply_rule(unseen_b["rule"], P["row"], pu, key, nb, oe_local=False)
                u = row_unseen[P["row"]]
                p = np.where(u, pu, p)
                pred = np.where(u, predu, pred)
        del X
        t_model = time.time() - tm
        pk = P["row"] * R + np.maximum(b_idx, 0)
        if len(np.unique(pk[~miss])) != int((~miss).sum()):
            errors.append("duplicate candidate ids inside a row")
        bad_prefix = len(P["cand_id"]) - int(pc.sum(pc.or_(pc.starts_with(P["cand_id"], "S2-"),
                                                           pc.starts_with(P["cand_id"], "S3-"))).as_py() or 0)
        if bad_prefix:
            errors.append(f"{bad_prefix} candidate ids without S2-/S3- prefix")
        tp_ = ef.group_reduce(P["row"], p, nb, "max", 0.0, is_sorted=True)
        row_top_p[base:base + nb] = tp_
        if sw is not None:
            nzp = np.flatnonzero((p > 0) & ~miss)
            sw.write_table(pa.table({"row": pa.array((P["row"][nzp] + base).astype(np.int32)),
                                     "cand": pc.cast(P["cand_id"].take(pa.array(nzp)), pa.string()),
                                     "p": pa.array(p[nzp].astype(np.float32))}))
        acc_row.append(P["row"][pred] + base)
        acc_key.append(key[pred])
        acc_p.append(p[pred].astype(np.float32))

        lists = c["candidate_entity_ids"].combine_chunks()
        if tk:
            lists = pc.list_slice(lists, 0, tk)
        lines = pc.binary_join_element_wise(c["source1_entity_id"].combine_chunks(),
                                            pc.binary_join(lists, ","), "\t")
        fc.write("\n".join(lines.to_pylist()) + "\n")
        ef.log(f"batch rows {base:,}-{base + nb:,}: {len(p):,} pairs, {int(pred.sum()):,} accepted "
               f"({time.time() - tb:.0f}s = features {t_feat:.0f} + model {t_model:.0f} + "
               f"checks/write {time.time() - tb - t_feat - t_model:.0f})")
        base += nb
        del P, p, pred, lines, c
    absent = pc.filter(all_ids, pc.invert(pc.is_in(all_ids, value_set=s1_seen))).to_pylist()
    for sid in absent:
        fc.write(f"{sid}\t\n")
    fc.close()
    if sw is not None:
        sw.close()
        pq.write_table(pa.table({"row": pa.array(np.arange(n_s1, dtype=np.int32)), "s1_id": s1_seen,
                                 "country": pa.array(list(ctry))}),
                       os.path.join(a.out_dir, "scores_rows.parquet"))
        with open(os.path.join(a.out_dir, "scores_meta.json"), "w") as f:
            json.dump({"model_dir": a.model_dir, "rule": main_b["rule"], "owner_exclusive": oe_global,
                       "absent_s1": absent, "p_floor": bool(meta.get("p_floor"))}, f)
        ef.log(f"saved pair scores to {a.out_dir}/scores.parquet (redecide.py can re-apply a rule in minutes)")

    row = np.concatenate(acc_row) if acc_row else np.zeros(0, np.int64)
    key = np.concatenate(acc_key) if acc_key else np.zeros(0, np.int64)
    p = np.concatenate(acc_p) if acc_p else np.zeros(0, np.float32)

    # ---------------- contention (before owner-exclusive)
    D = []

    def dlog(msg=""):
        ef.log(msg)
        D.append(msg)

    pre = np.bincount(row, minlength=n_s1)
    kk = np.where(key >= 0, key, 0)
    claims = np.bincount(kk, minlength=R)[kk]
    contested = (claims >= 2) & (key >= 0)
    rc = ctry_i[row]
    dlog("\n==================== CONTENTION & CONFIDENCE BY COUNTRY (no labels needed) ====================")
    dlog(f"{'country':8s} {'rows':>9} {'acc/row pre-OE':>14} {'contested pairs':>16} {'rows touched':>13} "
         f"{'top p>=0.99':>12} {'top p<0.5':>10}")
    for ci, cn in enumerate(countries):
        m = ctry_i == ci
        am = rc == ci
        touched = np.zeros(n_s1, bool)
        touched[row[contested]] = True
        dlog(f"{cn:8s} {m.sum():9,} {pre[m].mean():14.2f} {contested[am].mean() * 100 if am.any() else 0:15.1f}% "
             f"{touched[m].mean() * 100:12.1f}% {(row_top_p[m] >= 0.99).mean() * 100:11.1f}% "
             f"{(row_top_p[m] < 0.5).mean() * 100:9.1f}%")
    dlog("A contested pair = an S2/S3 record accepted for 2+ S1 rows; all but one of those claims are wrong.")
    dlog("A country far above the others here is where precision is leaking.")

    # ---------------- feature drift on rank-0 candidates
    mean = dsum / np.maximum(dn, 1)[:, None]
    seen = [cix[c] for c in countries if c in train_countries] or \
           [cix[c] for c in countries if c in ("US", "India")]
    if seen:
        ns = dn[seen].sum()
        mu = dsum[seen].sum(0) / max(ns, 1)
        sd = np.sqrt(np.maximum(dsq[seen].sum(0) / max(ns, 1) - mu ** 2, 1e-12))
        dlog("\n==================== FEATURE DRIFT (rank-0 candidate of each row) ====================")
        dlog(f"z = (country mean - training-country mean) / training-country std; reference countries: "
             f"{[str(countries[i]) for i in seen]}. Between-training-country z shows the normal spread.")
        for ci, cn in enumerate(countries):
            z = (mean[ci] - mu) / sd
            top = np.argsort(-np.abs(z))[:12]
            dlog(f"  {cn:8s}: " + ", ".join(f"{ef.FEATURES[j]}={z[j]:+.2f}" for j in top))

    # ---------------- contested examples
    ex = []
    rng = np.random.RandomState(0)
    nos, adr, rid = rec["nosuf"], rec["addr"], rec["ids"]
    txt = lambda i: f"{rid[i].as_py()} | {nos[i].as_py()} | {adr[i].as_py()}" if i >= 0 else "(missing)"
    if contested.any():
        order = np.lexsort((-p, key))
        ko = key[order]
        for ci, cn in enumerate(countries):
            keys_c = np.unique(key[contested & (rc == ci)])
            if not len(keys_c):
                continue
            pick = rng.choice(keys_c, min(a.contested_examples, len(keys_c)), replace=False)
            ex.append(f"\n########## {cn}: {len(keys_c):,} contested records (showing {len(pick)}) ##########")
            for k_ in pick:
                lo, hi = np.searchsorted(ko, k_), np.searchsorted(ko, k_, side="right")
                ex.append(f"\nRECORD {txt(int(k_))}")
                for j, jj in enumerate(order[lo:hi]):
                    tag = "KEEP" if (j == 0 and oe_global) else ("drop" if oe_global else "    ")
                    ex.append(f"   {tag} p={p[jj]:.3f} [{ctry[row[jj]]}] S1 {txt(int(s1_rec_all[row[jj]]))}")
    with open(os.path.join(a.out_dir, "contested_examples.txt"), "w") as f:
        f.write("\n".join(ex) if ex else "no contested records\n")

    # ---------------- global owner-exclusive + matching_results.tsv
    if oe_global and len(p):
        keep = ef.owner_exclusive(np.ones(len(p), bool), p, key)
        dlog(f"\nowner-exclusive removed {int((~keep).sum()):,} of {len(p):,} accepted pairs")
        row, key, p = row[keep], key[keep], p[keep]
    order = np.lexsort((-p, row))
    row, key = row[order], key[order]
    if (key < 0).any():
        errors.append("accepted a pair whose candidate id is missing from the TSVs")
        good = key >= 0
        row, key = row[good], key[good]
    mids = rec["ids"].take(pa.array(key)).to_pylist()
    npred = np.bincount(row, minlength=n_s1)
    starts = np.concatenate([[0], np.cumsum(npred)[:-1]])
    with open(os.path.join(a.out_dir, "matching_results.tsv"), "w") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for i, sid in enumerate(s1_list):
            m = mids[starts[i]:starts[i] + npred[i]]
            fm.write(f"{sid}\t{','.join(m)}\n")
        for sid in absent:
            fm.write(f"{sid}\t\n")
    if len(set(s1_list) - set(s1_country)):
        errors.append("S1 ids in candidates that are not in the test source1 file")
    dlog("FORMAT CHECKS: " + ("PASS" if not errors else f"FAIL: {sorted(set(errors))}"))

    dlog(f"validation reference: empty-rate {meta['val_pred_empty_rate'] * 100:.1f}%, "
         f"avg matches/row {meta['val_avg_pred_per_row']:.2f}")
    for ci, cn in enumerate(countries):
        m = ctry_i == ci
        dlog(f"  {cn:8s}: {m.sum():9,} rows | empty-rate {(npred[m] == 0).mean() * 100:5.1f}% | "
             f"avg matches/row {npred[m].mean():.2f}")
    if main_b.get("n_skipped"):
        dlog(f"prefilter: big model skipped on {main_b['n_skipped']:,} pairs (training-country rows only)")
    with open(os.path.join(a.out_dir, "diagnostics.txt"), "w") as f:
        f.write("\n".join(D) + "\n")
    ef.log(f"done in {time.time() - T0:.0f}s. Diagnostics: {a.out_dir}/diagnostics.txt, "
           f"{a.out_dir}/contested_examples.txt")


if __name__ == "__main__":
    main()

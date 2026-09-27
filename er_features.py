"""
Shared feature pipeline for the matcher: used by BOTH train_matcher_v2.py and
predict_matches.py, so train and test features are computed by the exact same
code (the classic way to silently lose points is a train/test feature skew).

Inputs: a v7 candidates parquet (blocking_recall_v7.py) + the raw TSVs.
Everything per-pair is vectorized or chunked across worker processes; strings
live in pyarrow arrays (no per-string Python objects shared across forks, so no
copy-on-write blowup), and each worker materializes only its own chunk.
"""
import multiprocessing as mp
import os
import re
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from normalize import clean_name, clean_address, numeric_suffix_overlap
from build_translit_dict import is_native

FAMILY_BITS = ["squash", "rare_name", "rare_ngram", "addr_num", "addr_pin", "rare_addr",
               "addr_pair", "name_pair", "tfidf"]

# ---- feature names, grouped (groups drive the ablation report) ----
FEATURE_GROUPS = {
    "blocking": ["rank", "log_rank", "rank_score", "coarse_score", "agreement", "jacc_name_tok",
                 "jacc_addr_tok", "is_s3"] + [f"fam_{b}" for b in FAMILY_BITS],
    "name_str": ["n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw_sq", "n_len_ratio",
                 "n_first_tok_eq", "n_prefix4_eq", "n_ntok_diff",
                 # extra tokens: sibling entities ADD a word at the end ("x public", "x holdings"),
                 # noise adds one at the start ("smt x", "shri x") -- token_set_ratio sees neither
                 "n_xa", "n_xb", "n_xa_first", "n_xa_last", "n_xb_first", "n_xb_last"],
    "addr_str": ["a_ratio", "a_tset", "a_len_ratio"],     # a_partial dropped: 40% of rapidfuzz time, ~0 gain
    "numeric": ["num_jacc", "num_first_eq", "num_any_eq", "num_suffix", "pin_eq", "pin_conflict",
                "num_both",
                # corruption of the SAME number (400->40, 34->0034) vs a NEARBY different number
                # (4747 vs 4750: sibling entity next door)
                "num_prefix", "num_strip0_eq", "num_min_reldiff"],
    "flags": ["either_domain", "either_alias", "either_phone", "a_empty_either", "a_empty_both",
              "either_native"],
    "group": ["g_n", "g_rs_gap", "g_rs_top2_gap", "g_ntset_gap", "g_nratio_gap", "g_aset_gap",
              "g_rank_ntset", "g_n_strong"],
    "row": ["log_n_before_prune", "tfidf_ran"] + [f"log_hits_{b}" for b in FAMILY_BITS],
    # competing S1 rows that list the same record (blocking_recall.py reverse columns): does ANOTHER
    # S1 entity fit this record better? Needs parquets built with --sample-s1 0.
    "reverse": ["rev_n_log", "rev_r_other", "rev_r_gap", "rev_jn_other", "rev_jn_gap",
                "rev_ja_other", "rev_ja_gap", "rev_top_r"],
    # rare-word-weighted similarity: city/region/"rue"/"street" count for little, a street name or
    # rare name word counts for a lot; *_miss_* = rarest word present on one side only
    "idf": ["a_idf_jacc", "a_idf_cov", "a_idf_miss_a", "a_idf_miss_b",
            "n_idf_jacc", "n_idf_cov", "n_idf_miss_a", "n_idf_miss_b",
            # typo-tolerant: a token also counts as matched if Jaro-Winkler >= 0.88 to an unmatched
            # token on the other side ('bouriloln' ~ 'bourillon'), so a remaining rare miss is a
            # genuinely different street/name word ('genets' vs 'oyats')
            "a_soft_cov", "a_soft_miss_a", "a_soft_miss_b", "n_soft_cov", "n_soft_miss_a", "n_soft_miss_b"],
    # how many S1 entities / S2+S3 records in the whole split share this exact (squashed) name:
    # a blank-address record with a unique name is almost surely a match, one shared by 5 S1s is not
    "namefreq": ["nf_s1_cand", "nf_rec_cand", "nf_s1_own", "nf_rec_own", "nf_rec_per_s1"],
}
FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]
_STR_FEATS = FEATURE_GROUPS["name_str"] + FEATURE_GROUPS["addr_str"] + FEATURE_GROUPS["numeric"] + \
    FEATURE_GROUPS["idf"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)



# --------------------------------------------------------------------------- group reductions
def group_reduce(keys, vals, n, how="max", init=0.0, is_sorted=False):
    """per-key max/min of vals (keys in [0, n)). Sort + reduceat: fast on every numpy version
    (np.maximum.at is ~100x slower on numpy < 1.25)."""
    out = np.full(n, init, dtype=np.float64)
    if not len(keys):
        return out
    if is_sorted:
        k, v = keys, vals
    else:
        o = np.argsort(keys, kind="stable")
        k, v = keys[o], vals[o]
    st = np.flatnonzero(np.concatenate([[True], k[1:] != k[:-1]]))
    f = np.maximum if how == "max" else np.minimum
    out[k[st]] = f.reduceat(np.asarray(v, np.float64), st)
    return out

# --------------------------------------------------------------------------- raw records
def read_tsv_arrow(path):
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").rstrip("\r").split("\t")
    try:
        return pacsv.read_csv(
            path, read_options=pacsv.ReadOptions(block_size=64 << 20),
            parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
            convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in header},
                                                 strings_can_be_null=False,
                                                 quoted_strings_can_be_null=False))
    except Exception as e:
        import pandas as pd
        log(f"  pyarrow parse failed ({e}); pandas fallback for {path}")
        return pa.Table.from_pandas(pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                                                na_filter=False, quoting=3), preserve_index=False)


def read_records(data_dir, split, needed_ids: pa.Array) -> pa.Table:
    parts = []
    vs = pc.unique(needed_ids)
    for s in (1, 2, 3):
        t = read_tsv_arrow(os.path.join(data_dir, split, f"{split}_source{s}.tsv"))
        t = t.filter(pc.is_in(t["entity_id"], value_set=vs))
        parts.append(t.select(["entity_id", "business_name", "business_address", "country"]))
    return pa.concat_tables(parts).combine_chunks()


# --------------------------------------------------------------------------- normalization
_SQ = re.compile(r"[^a-z0-9]")
_NORM = {}


def _latinize(text, td):
    if not is_native(text):
        return text
    return " ".join(td.get(raw.casefold().strip(".,()[]{}'\""), raw) for raw in text.split())


def _norm_chunk(args):
    names, addrs = args
    td = _NORM["translit"]
    out = {k: [] for k in ("nosuf", "sq", "addr", "nums", "al_nosuf", "al_sq")}
    flags = np.zeros((len(names), 5), np.bool_)
    for i, (n, a) in enumerate(zip(names, addrs)):
        ni = clean_name(_latinize(n, td))
        ai = clean_address(a)
        v = ni["variants"][0]
        out["nosuf"].append(v["no_suffix"])
        out["sq"].append(v["squashed"])
        v2 = ni["variants"][1] if len(ni["variants"]) > 1 else None
        out["al_nosuf"].append(v2["no_suffix"] if v2 else "")
        out["al_sq"].append(v2["squashed"] if v2 else "")
        out["addr"].append(ai["casefolded"])
        out["nums"].append(" ".join(ai["numeric_tokens"]))
        flags[i] = (ni["is_domain_like"], ni["had_alias_marker"], ni["had_phone_suffix"],
                    ai["is_empty"], is_native(n))
    return out, flags


def normalize_records(tbl: pa.Table, translit: dict, workers: int, chunk=50_000) -> dict:
    n = tbl.num_rows
    _NORM["translit"] = translit
    names, addrs = tbl["business_name"], tbl["business_address"]
    tasks = ((names.slice(s, chunk).to_pylist(), addrs.slice(s, chunk).to_pylist())
             for s in range(0, n, chunk))
    if workers <= 1 or n < 2 * chunk:
        parts = [_norm_chunk(t) for t in tasks]
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            parts = list(pool.imap(_norm_chunk, tasks))
    rec = {"ids": tbl["entity_id"].combine_chunks(), "country": tbl["country"].combine_chunks()}
    for k in ("nosuf", "sq", "addr", "nums", "al_nosuf", "al_sq"):
        rec[k] = pa.array([x for p in parts for x in p[0][k]], type=pa.large_string())
    fl = np.concatenate([p[1] for p in parts]) if parts else np.zeros((0, 5), bool)
    for j, k in enumerate(("is_domain", "alias", "phone", "addr_empty", "native")):
        rec[k] = fl[:, j].copy()
    return tokenize_records(rec)



# --------------------------------------------------------------------------- per-record token arrays
# Every record is tokenized ONCE into integer ids (records recur in up to ~200 pairs each), so the
# per-pair set/number/idf features below are pure numpy gathers + sorted-key membership tests.
_NUM_CAP = 18      # digits kept for numeric values (int64)


def _tokenize(arr, split_regex=None):
    """-> (ids int32 flat, offsets int64, vocabulary pa.Array); empty tokens dropped (= str.split())."""
    if split_regex:
        arr = pc.replace_substring_regex(arr, split_regex, " ")
    lists = pc.utf8_split_whitespace(arr)
    flat = pc.list_flatten(lists)
    parent = pc.list_parent_indices(lists).to_numpy(zero_copy_only=False)
    keep = pc.greater(pc.utf8_length(flat), 0)
    flat = flat.filter(keep)
    parent = parent[keep.to_numpy(zero_copy_only=False)]
    d = pc.dictionary_encode(flat).combine_chunks() if isinstance(flat, pa.ChunkedArray) else pc.dictionary_encode(flat)
    ids = d.indices.to_numpy(zero_copy_only=False).astype(np.int32)
    off = np.zeros(len(arr) + 1, np.int64)
    np.cumsum(np.bincount(parent, minlength=len(arr)), out=off[1:])
    return ids, off, d.dictionary


def _dedup_sorted(ids, off, V):
    """per-record sorted unique ids (same CSR layout)."""
    n = len(off) - 1
    parent = np.repeat(np.arange(n, dtype=np.int64), np.diff(off))
    key = np.unique(parent * V + ids)
    p2 = key // V
    o2 = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(p2, minlength=n), out=o2[1:])
    return (key % V).astype(np.int32), o2


def _idf(ids_set, V, n_docs):
    df = np.bincount(ids_set, minlength=V).astype(np.float64)
    return (np.log((n_docs + 1) / (df + 1)) / np.log(n_docs + 1)).astype(np.float32)


def tokenize_records(rec):
    t = time.time()
    n = len(rec["ids"])
    ids, off, voc = _tokenize(rec["nosuf"])
    V = max(len(voc), 1)
    rec["nm_ids"], rec["nm_off"] = ids, off                       # name tokens, original order
    rec["nm_sid"], rec["nm_soff"] = _dedup_sorted(ids, off, V)     # name token sets
    rec["nm_idf"], rec["nm_V"] = _idf(rec["nm_sid"], V, n), V
    rec["nm_voc"], rec["nm_vlen"] = voc, pc.utf8_length(voc).to_numpy(zero_copy_only=False).astype(np.int64)
    rec["nm_vocs"] = np.array(voc.to_pylist(), dtype=object)     # built ONCE here, shared by forked workers
    ids, off, voc = _tokenize(rec["addr"], r"[^\pL\pN]+")
    V = max(len(voc), 1)
    rec["ad_sid"], rec["ad_soff"] = _dedup_sorted(ids, off, V)     # address word+number sets
    rec["ad_idf"], rec["ad_V"] = _idf(rec["ad_sid"], V, n), V
    rec["ad_voc"], rec["ad_vlen"] = voc, pc.utf8_length(voc).to_numpy(zero_copy_only=False).astype(np.int64)
    rec["ad_vocs"] = np.array(voc.to_pylist(), dtype=object)
    ids, off, voc = _tokenize(rec["nums"])
    rec["nu_ids"], rec["nu_off"], rec["nu_V"] = ids, off, max(len(voc), 1)
    lens = pc.utf8_length(voc).to_numpy(zero_copy_only=False).astype(np.int64)
    tail = pc.utf8_slice_codeunits(voc, -_NUM_CAP) if len(voc) else voc
    rec["nu_len"] = lens
    rec["nu_val"] = pc.cast(tail, pa.int64()).to_numpy(zero_copy_only=False) if len(voc) else np.zeros(0, np.int64)
    rec["sq_len"] = pc.utf8_length(rec["sq"]).to_numpy(zero_copy_only=False).astype(np.int64)
    rec["addr_len"] = pc.utf8_length(rec["addr"]).to_numpy(zero_copy_only=False).astype(np.int64)
    p4 = pc.dictionary_encode(pc.utf8_slice_codeunits(rec["sq"], 0, 4)).indices.to_numpy(zero_copy_only=False)
    rec["sq_p4"] = np.where(rec["sq_len"] >= 4, p4, -1).astype(np.int64)
    log(f"  tokenized {n:,} records in {time.time() - t:.0f}s (name vocab {rec['nm_V']:,}, "
        f"address vocab {rec['ad_V']:,}, number vocab {rec['nu_V']:,})")
    return rec



# --------------------------------------------------------------------------- name frequency
def _sq_chunk(names):
    td = _NORM["translit"]
    return [clean_name(_latinize(n, td))["variants"][0]["squashed"] for n in names]


def attach_name_freq(rec, data_dir, split, translit, workers, chunk=100_000):
    """rec['nf_s1'] / rec['nf_rec'] = how many S1 rows / S2+S3 records of the WHOLE split (same country)
    have exactly this record's squashed name. Uses the same clean_name path as rec['sq']."""
    t = time.time()
    _NORM["translit"] = translit
    keys = {}
    for grp, srcs in (("s1", (1,)), ("rec", (2, 3))):
        parts = []
        for sidx in srcs:
            tb = read_tsv_arrow(os.path.join(data_dir, split, f"{split}_source{sidx}.tsv")).select(
                ["business_name", "country"])
            names = tb["business_name"].combine_chunks()
            tasks = (names.slice(i, chunk).to_pylist() for i in range(0, len(names), chunk))
            if workers <= 1:
                sq = [x for tk in tasks for x in _sq_chunk(tk)]
            else:
                with mp.get_context("fork").Pool(workers) as pool:
                    sq = [x for part in pool.imap(_sq_chunk, tasks) for x in part]
            parts.append(pc.binary_join_element_wise(tb["country"].combine_chunks(),
                                                     pa.array(sq, type=pa.string()), "\x1f"))
            del tb, names, sq
        vc = pc.value_counts(pa.concat_arrays(parts))
        keys[grp] = (vc.field("values"), vc.field("counts").to_numpy(zero_copy_only=False))
    rk = pc.binary_join_element_wise(pc.cast(rec["country"], pa.string()), pc.cast(rec["sq"], pa.string()), "\x1f")
    for grp, (vals, cnts) in keys.items():
        ix = pc.index_in(rk, value_set=vals).to_numpy(zero_copy_only=False)
        ix = np.asarray(ix, dtype=np.float64)
        ok = ~np.isnan(ix)
        out = np.zeros(len(ix), np.float32)
        out[ok] = cnts[ix[ok].astype(np.int64)]
        rec[f"nf_{grp}"] = out
    log(f"  name-frequency tables over the whole {split} split in {time.time() - t:.0f}s")
    return rec


def rec_index(rec, ids: pa.Array) -> np.ndarray:
    """Positions of `ids` in the record table (-1 if absent)."""
    ix = pc.index_in(ids, value_set=rec["ids"]).to_numpy(zero_copy_only=False)
    ix = np.asarray(ix, dtype=np.float64)
    return np.where(np.isnan(ix), -1, ix).astype(np.int64)


def load_candidates(paths, max_s1=0, rng=None, batch_rows=200_000):
    """Read candidate parquet(s) keeping only a random sample of max_s1 S1 rows, WITHOUT ever
    holding the full files in memory (the full train parquets are millions of rows x topk).
    Picks the same rows as pa.concat_tables(read_table(...)).take(sorted(rng.choice(n_all, max_s1))).
    Columns missing from some files are dropped (with a warning); types follow the first file.
    -> (table, n_all, n_duplicate_s1_ids)"""
    files = [pq.ParquetFile(p) for p in paths]
    n_all = sum(f.metadata.num_rows for f in files)
    schemas = [f.schema_arrow for f in files]
    common = [c for c in schemas[0].names if all(c in s.names for s in schemas)]
    extra = sorted({c for s in schemas for c in s.names} - set(common))
    if extra:
        log(f"  WARNING: columns not present in every candidates file, ignored: {extra} "
            f"(regenerate all files with the same blocking_recall.py if these matter)")
    target = pa.schema([schemas[0].field(c) for c in common])
    ids = pa.concat_arrays([pq.read_table(p, columns=["source1_entity_id"])["source1_entity_id"].combine_chunks()
                            for p in paths])
    dup = n_all - len(pc.unique(ids))
    del ids
    take = None
    if max_s1 and max_s1 < n_all:
        take = np.sort((rng or np.random).choice(n_all, max_s1, replace=False))
    parts, base = [], 0
    for f in files:
        for b in f.iter_batches(batch_size=batch_rows, columns=common):
            t = pa.Table.from_batches([b])
            if take is not None:
                lo, hi = np.searchsorted(take, [base, base + t.num_rows])
                t = t.take(pa.array(take[lo:hi] - base)) if hi > lo else None
            if t is not None:
                parts.append(t if t.schema.equals(target) else t.cast(target))
            base += b.num_rows
    tbl = pa.concat_tables(parts).combine_chunks() if parts else target.empty_table()
    return tbl, n_all, dup


# --------------------------------------------------------------------------- candidates -> pairs
REQUIRED_V7 = ["candidate_entity_ids", "candidate_agreement", "candidate_coarse_score",
               "candidate_rank_score", "n_before_prune"]


def explode_candidates(cand: pa.Table, topk_use=None) -> dict:
    """Flatten per-S1 candidate lists into per-pair numpy arrays (row = index into `cand`)."""
    missing = [c for c in REQUIRED_V7 if c not in cand.column_names]
    if missing:
        raise SystemExit(f"candidates parquet lacks {missing} -- regenerate it with blocking_recall_v7.py")
    lists = cand["candidate_entity_ids"].combine_chunks()
    off = lists.offsets.to_numpy().astype(np.int64)
    lens = np.diff(off)
    if topk_use:
        lens = np.minimum(lens, topk_use)
    n_rows = cand.num_rows
    row = np.repeat(np.arange(n_rows, dtype=np.int64), lens)
    start = np.repeat(off[:-1], lens)
    rank = np.arange(len(row), dtype=np.int64) - np.repeat(np.cumsum(lens) - lens, lens)
    flat_ix = start + rank

    def flat(col, dtype):
        if col not in cand.column_names:
            return None
        v = cand[col].combine_chunks().values.to_numpy(zero_copy_only=False)
        return v[flat_ix].astype(dtype)

    P = {
        "row": row, "rank": rank,
        "cand_id": lists.values.take(pa.array(flat_ix)),
        "agreement": flat("candidate_agreement", np.float32),
        "coarse_score": flat("candidate_coarse_score", np.float32),
        "rank_score": flat("candidate_rank_score", np.float32),
        "mask": flat("candidate_family_mask", np.int32),
        "jn": flat("candidate_jaccard_name", np.float32),
        "ja": flat("candidate_jaccard_addr", np.float32),
        "rev_n": flat("candidate_rev_n", np.float32),
        "rev_r": flat("candidate_rev_rank_score", np.float32),
        "rev_jn": flat("candidate_rev_jaccard_name", np.float32),
        "rev_ja": flat("candidate_rev_jaccard_addr", np.float32),
        "row_lens": lens,
    }
    if P["rev_n"] is None:
        log("  WARNING: no candidate_rev_* columns -- reverse features will be 0. Rebuild candidates with the "
            "current blocking_recall.py and --sample-s1 0.")
    if P["mask"] is None:
        log("  WARNING: no candidate_family_mask/jaccard columns (v7.0 parquet) -- those features "
            "will be 0. Regenerate with the current blocking_recall_v7.py for full features.")
    return P


# --------------------------------------------------------------------------- string features
_R = {}
_TOK_KEYS = ("nm_ids", "nm_off", "nm_sid", "nm_soff", "nm_idf", "nm_V", "ad_sid", "ad_soff", "ad_idf", "ad_V",
             "nm_voc", "nm_vlen", "ad_voc", "ad_vlen", "nm_vocs", "ad_vocs",
             "nu_ids", "nu_off", "nu_V", "nu_len", "nu_val", "sq_len", "addr_len", "sq_p4")



def _gather(ids, off, rows):
    """concatenated token lists of `rows` -> (tokens, per-row lengths, owning pair index)."""
    st = off[rows]
    ln = off[rows + 1] - st
    tot = int(ln.sum())
    if not tot:
        return np.zeros(0, ids.dtype), ln, np.zeros(0, np.int64)
    base = np.repeat(st - (np.cumsum(ln) - ln), ln)
    return ids[base + np.arange(tot)], ln, np.repeat(np.arange(len(rows), dtype=np.int64), ln)


def _member(keys_q, keys_sorted):
    if not len(keys_sorted):
        return np.zeros(len(keys_q), bool)
    s = np.searchsorted(keys_sorted, keys_q)
    s = np.minimum(s, len(keys_sorted) - 1)
    return keys_sorted[s] == keys_q


_SOFT_JW, _SOFT_MINLEN, _SOFT_CAP = 0.88, 3, 6


def _soft_best(voc, vlen, ga, pa_, una, gb, pb_, unb, m):
    """best Jaro-Winkler of every unmatched token against the OTHER side's unmatched tokens of the
    same pair (typo-tolerant matching: 'bouriloln' ~ 'bourillon'); tokens < 3 chars or beyond the
    first 6 unmatched per side are skipped. -> (best_a per token of ga, best_b per token of gb)"""
    from rapidfuzz import process
    from rapidfuzz.distance import JaroWinkler
    best_a, best_b = np.zeros(len(ga)), np.zeros(len(gb))
    ia = np.flatnonzero(una & (vlen[ga] >= _SOFT_MINLEN))
    ib = np.flatnonzero(unb & (vlen[gb] >= _SOFT_MINLEN))
    if not len(ia) or not len(ib):
        return best_a, best_b

    def capped(ix, pp):
        c = np.bincount(pp[ix], minlength=m)
        st = np.cumsum(c) - c
        rank = np.arange(len(ix)) - st[pp[ix]]
        ix = ix[rank < _SOFT_CAP]
        c = np.bincount(pp[ix], minlength=m)
        return ix, c, np.cumsum(c) - c
    ia, ca, sa = capped(ia, pa_)
    ib, cb, sb = capped(ib, pb_)
    cnt = ca * cb
    tot = int(cnt.sum())
    if not tot:
        return best_a, best_b
    pc_ = np.repeat(np.arange(m, dtype=np.int64), cnt)
    loc = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    xa = ia[sa[pc_] + loc // cb[pc_]]
    xb = ib[sb[pc_] + loc % cb[pc_]]
    ta, tb = ga[xa].astype(np.int64), gb[xb].astype(np.int64)
    ok = np.abs(vlen[ta] - vlen[tb]) <= 3                  # JW >= 0.88 needs similar lengths
    xa, xb, ta, tb = xa[ok], xb[ok], ta[ok], tb[ok]
    if not len(xa):
        return best_a, best_b
    # each distinct token pair is scored once (the S1 side's tokens recur across its ~200 candidates)
    key = ta * (int(vlen.shape[0]) + 1) + tb
    uk, inv = np.unique(key, return_inverse=True)
    vs = voc
    V1 = int(vlen.shape[0]) + 1
    jw_u = process.cpdist(vs[uk // V1].tolist(), vs[uk % V1].tolist(),
                          scorer=JaroWinkler.normalized_similarity, workers=1, dtype=np.float32)
    jw = jw_u[inv.ravel()]
    # xa is grouped (i-major, j-minor): contiguous per a-token; xb needs a sort
    st = np.flatnonzero(np.concatenate([[True], xa[1:] != xa[:-1]]))
    best_a[xa[st]] = np.maximum.reduceat(jw, st)
    best_b[:] = group_reduce(xb, jw, len(gb), "max", 0.0)
    return best_a, best_b




def _set_sims(sid, soff, idf, V, a, b, voc=None, vlen=None):
    """set overlap between each pair's token sets: counts, idf-weighted jaccard/coverage, rarest miss,
    and (with voc) the same after typo-tolerant token matching."""
    m = len(a)
    ga, la, pa_ = _gather(sid, soff, a)
    gb, lb, pb_ = _gather(sid, soff, b)
    ka, kb = pa_ * V + ga, pb_ * V + gb           # sorted: pairs ascending, ids sorted per record
    ina, inb = _member(ka, kb), _member(kb, ka)
    inter = np.bincount(pa_, weights=ina, minlength=m)
    wa_t, wb_t = idf[ga], idf[gb]
    wa = np.bincount(pa_, weights=wa_t, minlength=m)
    wb = np.bincount(pb_, weights=wb_t, minlength=m)
    wi = np.bincount(pa_, weights=wa_t * ina, minlength=m)
    miss_a = group_reduce(pa_, np.where(ina, 0.0, wa_t), m, "max", 0.0, is_sorted=True)
    miss_b = group_reduce(pb_, np.where(inb, 0.0, wb_t), m, "max", 0.0, is_sorted=True)
    den = wa + wb - wi
    jac = np.divide(wi, den, out=np.zeros(m), where=den > 0)
    mn = np.minimum(wa, wb)
    cov = np.divide(wi, mn, out=np.zeros(m), where=mn > 0)
    soft = None
    if voc is not None:
        ba, bb = _soft_best(voc, vlen, ga, pa_, ~ina, gb, pb_, ~inb, m)
        sa_w = np.where(ina, 1.0, np.where(ba >= _SOFT_JW, ba, 0.0))
        sb_w = np.where(inb, 1.0, np.where(bb >= _SOFT_JW, bb, 0.0))
        num = np.bincount(pa_, weights=wa_t * sa_w, minlength=m) + np.bincount(pb_, weights=wb_t * sb_w, minlength=m)
        dd = wa + wb
        s_cov = np.divide(num, dd, out=np.zeros(m), where=dd > 0)
        s_ma = group_reduce(pa_, np.where(sa_w > 0, 0.0, wa_t), m, "max", 0.0, is_sorted=True)
        s_mb = group_reduce(pb_, np.where(sb_w > 0, 0.0, wb_t), m, "max", 0.0, is_sorted=True)
        soft = (s_cov, s_ma, s_mb)
    return la, lb, inter, jac, cov, miss_a, miss_b, (ka, kb), soft


def _vec_feats(a, b, F, col):
    R = _R
    m = len(a)
    # ---- lengths / 4-char prefix (squashed name), address length
    lx, ly = R["sq_len"][a], R["sq_len"][b]
    mx = np.maximum(lx, ly)
    F[:, col["n_len_ratio"]] = np.divide(np.minimum(lx, ly), mx, out=np.zeros(m), where=mx > 0)
    F[:, col["n_prefix4_eq"]] = (lx >= 4) & (R["sq_p4"][a] == R["sq_p4"][b])
    la_, lb_ = R["addr_len"][a], R["addr_len"][b]
    mx = np.maximum(la_, lb_)
    F[:, col["a_len_ratio"]] = np.divide(np.minimum(la_, lb_), mx, out=np.zeros(m), where=mx > 0)

    # ---- name tokens: first-token equality, token-count diff, extra words and where they sit
    V = R["nm_V"]
    ga, na_, pa_ = _gather(R["nm_ids"], R["nm_off"], a)
    gb, nb_, pb_ = _gather(R["nm_ids"], R["nm_off"], b)
    both = (na_ > 0) & (nb_ > 0)
    fa_i = np.cumsum(na_) - na_                       # index of each pair's first token in ga
    fb_i = np.cumsum(nb_) - nb_
    fa = np.where(na_ > 0, ga[np.minimum(fa_i, max(len(ga) - 1, 0))] if len(ga) else -1, -1)
    fb = np.where(nb_ > 0, gb[np.minimum(fb_i, max(len(gb) - 1, 0))] if len(gb) else -2, -2)
    la_last = np.where(na_ > 0, ga[np.minimum(fa_i + na_ - 1, max(len(ga) - 1, 0))] if len(ga) else -1, -1)
    lb_last = np.where(nb_ > 0, gb[np.minimum(fb_i + nb_ - 1, max(len(gb) - 1, 0))] if len(gb) else -2, -2)
    F[:, col["n_first_tok_eq"]] = both & (fa == fb)
    F[:, col["n_ntok_diff"]] = np.abs(na_ - nb_)
    ls_a, ls_b, inter, jac, cov, ma, mb, (ka, kb), soft = _set_sims(
        R["nm_sid"], R["nm_soff"], R["nm_idf"], V, a, b, R["nm_vocs"], R["nm_vlen"])
    ar = np.arange(m, dtype=np.int64)
    F[:, col["n_xa"]] = np.where(both, ls_a - inter, 0)
    F[:, col["n_xb"]] = np.where(both, ls_b - inter, 0)
    F[:, col["n_xa_first"]] = both & ~_member(ar * V + fa, kb)
    F[:, col["n_xa_last"]] = both & ~_member(ar * V + la_last, kb)
    F[:, col["n_xb_first"]] = both & ~_member(ar * V + fb, ka)
    F[:, col["n_xb_last"]] = both & ~_member(ar * V + lb_last, ka)
    F[:, col["n_idf_jacc"]], F[:, col["n_idf_cov"]] = jac, cov
    F[:, col["n_idf_miss_a"]], F[:, col["n_idf_miss_b"]] = ma, mb
    F[:, col["n_soft_cov"]], F[:, col["n_soft_miss_a"]], F[:, col["n_soft_miss_b"]] = soft

    # ---- address token sets, idf-weighted
    _, _, _, jac, cov, ma, mb, _, soft = _set_sims(R["ad_sid"], R["ad_soff"], R["ad_idf"], R["ad_V"], a, b,
                                                    R["ad_vocs"], R["ad_vlen"])
    F[:, col["a_idf_jacc"]], F[:, col["a_idf_cov"]] = jac, cov
    F[:, col["a_idf_miss_a"]], F[:, col["a_idf_miss_b"]] = ma, mb
    F[:, col["a_soft_cov"]], F[:, col["a_soft_miss_a"]], F[:, col["a_soft_miss_b"]] = soft

    # ---- numbers (original order, duplicates kept -- same semantics as the old per-pair loop)
    V = R["nu_V"]
    ga, na_, pa_ = _gather(R["nu_ids"], R["nu_off"], a)
    gb, nb_, pb_ = _gather(R["nu_ids"], R["nu_off"], b)
    both = (na_ > 0) & (nb_ > 0)
    F[:, col["num_both"]] = both
    if not both.any():
        F[:, col["num_min_reldiff"]] = 0.0
        return
    fa_i = np.cumsum(na_) - na_
    fb_i = np.cumsum(nb_) - nb_
    fa = ga[np.minimum(fa_i, len(ga) - 1)] if len(ga) else np.zeros(m, np.int32)
    fb = gb[np.minimum(fb_i, len(gb) - 1)] if len(gb) else np.zeros(m, np.int32)
    F[:, col["num_first_eq"]] = both & (fa == fb)
    # sets: jaccard, any equal, pin (>=5 digits) agreement
    ua = np.unique(pa_ * V + ga)
    ub = np.unique(pb_ * V + gb)
    ia = _member(ua, ub)
    na_s = np.bincount(ua // V, minlength=m)
    nb_s = np.bincount(ub // V, minlength=m)
    inter = np.bincount(ua // V, weights=ia, minlength=m)
    uni = na_s + nb_s - inter
    F[:, col["num_jacc"]] = np.where(both, np.divide(inter, uni, out=np.zeros(m), where=uni > 0), 0)
    F[:, col["num_any_eq"]] = both & (inter > 0)
    L = R["nu_len"]
    a5 = L[ua % V] >= 5
    b5 = L[ub % V] >= 5
    n5a = np.bincount(ua[a5] // V, minlength=m)
    n5b = np.bincount(ub[b5] // V, minlength=m)
    i5 = np.bincount(ua[a5] // V, weights=ia[a5], minlength=m)
    p5 = both & (n5a > 0) & (n5b > 0)
    F[:, col["pin_eq"]] = p5 & (i5 > 0)
    F[:, col["pin_conflict"]] = p5 & (i5 == 0)
    # all (x in A, y in B) combinations of each pair
    cnt = np.where(both, na_ * nb_, 0)
    tot = int(cnt.sum())
    pc_ = np.repeat(ar, cnt)
    loc = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    ib = loc % nb_[pc_]
    ia_ = loc // nb_[pc_]
    xi = fa_i[pc_] + ia_            # flat positions into ga / gb
    yi = fb_i[pc_] + ib
    x, y = ga[xi], gb[yi]
    lxn, lyn = L[x], L[y]
    vx, vy = R["nu_val"][x], R["nu_val"][y]
    p10 = 10 ** np.minimum(np.arange(_NUM_CAP + 1, dtype=np.int64), _NUM_CAP)
    ends_xy = (lyn <= lxn) & (vx % p10[np.minimum(lyn, _NUM_CAP)] == vy)
    ends_yx = (lxn <= lyn) & (vy % p10[np.minimum(lxn, _NUM_CAP)] == vx)
    suf = (lxn >= 2) & (lyn >= 2) & (ends_xy | ends_yx)
    hit_a = np.zeros(len(ga), bool)
    hit_b = np.zeros(len(gb), bool)
    hit_a[xi[suf]] = True
    hit_b[yi[suf]] = True
    ha = np.bincount(pa_, weights=hit_a & (L[ga] >= 2), minlength=m)
    hb = np.bincount(pb_, weights=hit_b & (L[gb] >= 2), minlength=m)
    shorter_a = na_ <= nb_
    F[:, col["num_suffix"]] = np.where(both, np.where(shorter_a, ha / np.maximum(na_, 1),
                                                      hb / np.maximum(nb_, 1)), 0)
    # first 8 numbers of each side: prefix relation, zero-padding, nearest different number
    c8 = (ia_ < 8) & (ib < 8) & (x != y)
    full = (lxn <= _NUM_CAP) & (lyn <= _NUM_CAP)
    d = np.clip(lxn - lyn, 0, _NUM_CAP)
    e = np.clip(lyn - lxn, 0, _NUM_CAP)
    sw_xy = (lyn <= lxn) & (vx // p10[d] == vy)
    sw_yx = (lxn <= lyn) & (vy // p10[e] == vx)
    pre = c8 & full & (lxn >= 2) & (lyn >= 2) & (sw_xy | sw_yx)
    z0 = c8 & full & (vx == vy)
    rd = c8 & ~z0 & (lxn <= 9) & (lyn <= 9)
    F[:, col["num_prefix"]] = np.bincount(pc_, weights=pre, minlength=m) > 0
    F[:, col["num_strip0_eq"]] = np.bincount(pc_, weights=z0, minlength=m) > 0
    best = np.ones(m)
    if rd.any():
        xv, yv = vx[rd].astype(np.float64), vy[rd].astype(np.float64)
        best = group_reduce(pc_[rd], np.abs(xv - yv) / np.maximum(np.maximum(xv, yv), 1), m, "min", 1.0,
                            is_sorted=True)
    F[:, col["num_min_reldiff"]] = np.where(both, best, 0)


def _pf_chunk(args):
    a, b = args
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import JaroWinkler
    R = _R
    ta, tb = pa.array(a), pa.array(b)
    g = lambda k, t: R[k].take(t).to_pylist()
    an, bn = g("nosuf", ta), g("nosuf", tb)
    asq, bsq = g("sq", ta), g("sq", tb)
    aad, bad = g("addr", ta), g("addr", tb)
    m = len(a)
    F = np.zeros((m, len(_STR_FEATS)), np.float32)
    c = lambda x, y, sc: process.cpdist(x, y, scorer=sc, workers=1, dtype=np.float32)
    col = {f: i for i, f in enumerate(_STR_FEATS)}
    F[:, col["n_ratio"]] = c(an, bn, fuzz.ratio) / 100
    F[:, col["n_tsort"]] = c(an, bn, fuzz.token_sort_ratio) / 100
    F[:, col["n_tset"]] = c(an, bn, fuzz.token_set_ratio) / 100
    F[:, col["n_partial"]] = c(an, bn, fuzz.partial_ratio) / 100
    F[:, col["n_jw_sq"]] = c(asq, bsq, JaroWinkler.normalized_similarity)
    F[:, col["a_ratio"]] = c(aad, bad, fuzz.ratio) / 100
    F[:, col["a_tset"]] = c(aad, bad, fuzz.token_set_ratio) / 100
    aal, bal = g("al_nosuf", ta), g("al_nosuf", tb)
    aas, bas = g("al_sq", ta), g("al_sq", tb)
    for i in range(m):
        if not (aal[i] or bal[i]):
            continue
        na_ = [an[i]] + ([aal[i]] if aal[i] else [])
        nb_ = [bn[i]] + ([bal[i]] if bal[i] else [])
        sa_ = [asq[i]] + ([aas[i]] if aas[i] else [])
        sb_ = [bsq[i]] + ([bas[i]] if bas[i] else [])
        for f, sc in (("n_ratio", fuzz.ratio), ("n_tsort", fuzz.token_sort_ratio),
                      ("n_tset", fuzz.token_set_ratio), ("n_partial", fuzz.partial_ratio)):
            F[i, col[f]] = max(sc(x, y) for x in na_ for y in nb_) / 100
        F[i, col["n_jw_sq"]] = max(JaroWinkler.normalized_similarity(x, y) for x in sa_ for y in sb_)
    _vec_feats(np.asarray(a, np.int64), np.asarray(b, np.int64), F, col)
    return F


def string_features(rec, a_idx, b_idx, workers, chunk=50_000):
    for k in ("nosuf", "sq", "addr", "nums", "al_nosuf", "al_sq") + _TOK_KEYS:
        _R[k] = rec[k]
    tasks = [(a_idx[s:s + chunk], b_idx[s:s + chunk]) for s in range(0, len(a_idx), chunk)]
    if workers <= 1 or len(tasks) <= 1:
        parts = [_pf_chunk(t) for t in tasks]
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            parts = list(pool.imap(_pf_chunk, tasks))
    return np.concatenate(parts) if parts else np.zeros((0, len(_STR_FEATS)), np.float32)


# --------------------------------------------------------------------------- group features
def _seg_max(v, starts, lens):
    out = np.full(len(lens), -np.inf, dtype=np.float64)
    nz = lens > 0
    out[nz] = np.maximum.reduceat(v, starts[nz])
    return out


def build_features(cand: pa.Table, P: dict, rec: dict, s1_rec_idx: np.ndarray, workers: int):
    """-> (X float32 [n_pairs, len(FEATURES)], cand_rec_idx int64)"""
    n = len(P["row"])
    X = np.zeros((n, len(FEATURES)), np.float32)
    fi = {f: i for i, f in enumerate(FEATURES)}
    b_idx = rec_index(rec, P["cand_id"])
    a_idx = s1_rec_idx[P["row"]]
    bad = (a_idx < 0) | (b_idx < 0)
    if bad.any():
        log(f"  WARNING: {bad.sum():,} pairs reference ids missing from the TSVs -- features zeroed")
    a_ok, b_ok = np.maximum(a_idx, 0), np.maximum(b_idx, 0)

    # blocking-side
    X[:, fi["rank"]] = P["rank"]
    X[:, fi["log_rank"]] = np.log1p(P["rank"])
    X[:, fi["rank_score"]] = P["rank_score"]
    X[:, fi["coarse_score"]] = P["coarse_score"]
    X[:, fi["agreement"]] = P["agreement"]
    if P["mask"] is not None:
        for j, b in enumerate(FAMILY_BITS):
            X[:, fi[f"fam_{b}"]] = (P["mask"] >> j) & 1
        X[:, fi["jacc_name_tok"]] = P["jn"]
        X[:, fi["jacc_addr_tok"]] = P["ja"]
    if P.get("rev_n") is not None:
        has = P["rev_n"] > 0
        X[:, fi["rev_n_log"]] = np.log1p(P["rev_n"])
        for f, own, oth in (("r", P["rank_score"], P["rev_r"]), ("jn", P["jn"], P["rev_jn"]),
                            ("ja", P["ja"], P["rev_ja"])):
            if own is None:
                continue
            X[:, fi[f"rev_{f}_other"]] = np.where(has, oth, -1.0)
            X[:, fi[f"rev_{f}_gap"]] = np.where(has, own - oth, 1.0)
        X[:, fi["rev_top_r"]] = np.where(has, P["rank_score"] >= P["rev_r"], 1.0)
    ids_s3 = pc.starts_with(P["cand_id"], "S3-").to_numpy(zero_copy_only=False)
    X[:, fi["is_s3"]] = ids_s3

    # string
    t = time.time()
    S = string_features(rec, a_ok, b_ok, workers)
    for j, f in enumerate(_STR_FEATS):
        X[:, fi[f]] = S[:, j]
    del S
    log(f"  string features for {n:,} pairs in {time.time() - t:.0f}s ({n / max(time.time() - t, 1e-9):,.0f}/s)")

    # flags
    for f, k in (("either_domain", "is_domain"), ("either_alias", "alias"), ("either_phone", "phone"),
                 ("a_empty_either", "addr_empty"), ("either_native", "native")):
        X[:, fi[f]] = rec[k][a_ok] | rec[k][b_ok]
    X[:, fi["a_empty_both"]] = rec["addr_empty"][a_ok] & rec["addr_empty"][b_ok]
    if "nf_s1" in rec:
        s1c, rcc = rec["nf_s1"][b_ok], rec["nf_rec"][b_ok]
        X[:, fi["nf_s1_cand"]] = np.log1p(s1c)
        X[:, fi["nf_rec_cand"]] = np.log1p(rcc)
        X[:, fi["nf_s1_own"]] = np.log1p(rec["nf_s1"][a_ok])
        X[:, fi["nf_rec_own"]] = np.log1p(rec["nf_rec"][a_ok])
        X[:, fi["nf_rec_per_s1"]] = rcc / np.maximum(s1c, 1)
    X[bad] = 0

    # group (pairs are contiguous per row, ordered by blocking rank)
    lens = P["row_lens"]
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
    rowmax = lambda v: _seg_max(v.astype(np.float64), starts, lens)[P["row"]]
    X[:, fi["g_n"]] = lens[P["row"]]
    rs = X[:, fi["rank_score"]]
    X[:, fi["g_rs_gap"]] = rowmax(rs) - rs
    rs64 = rs.astype(np.float64)
    top1 = _seg_max(rs64, starts, lens)
    order = np.lexsort((-rs64, P["row"]))
    rank_in_row = np.arange(n) - starts[P["row"][order]]
    second = np.full(len(lens), -np.inf)
    m2 = rank_in_row == 1
    second[P["row"][order][m2]] = rs64[order][m2]
    gap12 = np.where(np.isfinite(second), top1 - second, 1.0)
    X[:, fi["g_rs_top2_gap"]] = gap12[P["row"]]
    for f, src in (("g_ntset_gap", "n_tset"), ("g_nratio_gap", "n_ratio"), ("g_aset_gap", "a_tset")):
        v = X[:, fi[src]]
        X[:, fi[f]] = rowmax(v) - v
    v = X[:, fi["n_tset"]]
    order = np.lexsort((-v, P["row"]))
    r = np.empty(n, np.float32)
    r[order] = np.arange(n) - starts[P["row"][order]]
    X[:, fi["g_rank_ntset"]] = r
    strong = (X[:, fi["n_tset"]] >= 0.9).astype(np.float64)
    X[:, fi["g_n_strong"]] = np.bincount(P["row"], weights=strong, minlength=len(lens))[P["row"]]

    # row-level from blocking
    X[:, fi["log_n_before_prune"]] = np.log1p(
        cand["n_before_prune"].to_numpy().astype(np.float64))[P["row"]]
    if "tfidf_ran" in cand.column_names:
        X[:, fi["tfidf_ran"]] = cand["tfidf_ran"].to_numpy(zero_copy_only=False).astype(np.float32)[P["row"]]
    for b in FAMILY_BITS:
        c = f"hits_{b}"
        if c in cand.column_names:
            X[:, fi[f"log_hits_{b}"]] = np.log1p(cand[c].to_numpy().astype(np.float64))[P["row"]]

    bad_vals = ~np.isfinite(X)
    if bad_vals.any():
        log(f"  WARNING: {bad_vals.sum():,} non-finite feature values set to 0")
        X[bad_vals] = 0
    return X, b_idx


# --------------------------------------------------------------------------- decision rule
class Decider:
    """Row-level decision rule, precomputed once so a threshold grid is cheap.
    A row predicts nothing unless its best candidate has p >= t_top; then it predicts its best
    candidate plus every other candidate with p >= t_extra. With owner_exclusive, each S2/S3
    record is then kept only for the S1 row claiming it with the highest p (every real S2/S3
    record belongs to at most one S1). `key` identifies the S2/S3 record of each pair."""

    def __init__(self, row, p, key, n_rows):
        best = group_reduce(row, p, n_rows, "max", -1.0)
        self.p = p
        self.bestr = best[row]
        self.is_top = p >= self.bestr
        self.o = np.lexsort((-p, key))
        self.ko = key[self.o]

    def __call__(self, t_top, t_extra, owner_exclusive):
        pred = (self.bestr >= t_top) & (self.is_top | (self.p >= t_extra))
        if owner_exclusive:
            po = pred[self.o]
            idx, k = self.o[po], self.ko[po]
            first = np.ones(len(idx), bool)
            first[1:] = k[1:] != k[:-1]
            pred = np.zeros(len(pred), bool)
            pred[idx[first]] = True
        return pred


def decide(row, p, key, n_rows, t_top, t_extra, owner_exclusive):
    return Decider(row, p, key, n_rows)(t_top, t_extra, owner_exclusive)


def row_f_half(tp, npred, ntrue):
    P = np.divide(tp, npred, out=np.zeros(len(tp)), where=npred > 0)
    R = np.divide(tp, ntrue, out=np.zeros(len(tp)), where=ntrue > 0)
    d = 0.25 * P + R
    F = np.divide(1.25 * P * R, d, out=np.zeros(len(tp)), where=d > 0)
    F = np.where((ntrue == 0) & (npred == 0), 1.0, F)
    return F


# =========================================================================== stage 2
# Stage 1 scores every (S1, candidate) pair on its own. But the S2/S3 records that
# truly match one S1 are copies of each other, so once stage 1 is confident about
# SOME of a row's candidates, a further candidate that looks like those confident
# ones is likely a match too (recovers partial_recall_matcher), and one that looks
# like none of them is likely a false extra (cuts partial_extra_fp). Stage 2 adds,
# per pair, the stage-1 probability landscape of its row plus its similarity to
# the row's top-probability "anchor" records. Only pairs with p1 >= tau0 go
# through stage 2; the rest keep p1 (they are far below any threshold).
CONS_FEATS = ["p1", "p1_logit", "p_rank", "p_max_row", "p_second_row", "p_sum_row", "n50_row",
              "p_gap_top", "p_ratio", "is_anchor0",
              "o_p", "o_ntset", "o_nratio", "o_atset", "o_num", "o_jw",
              "sup_name", "sup_addr", "sup_both", "sup_cnt",
              "n_conf_same_src", "n_conf_other_src"]
FEATURES2 = FEATURES + CONS_FEATS


def _sim_chunk(args):
    a, b = args
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import JaroWinkler
    R = _R
    ta, tb = pa.array(a), pa.array(b)
    g = lambda k, t: R[k].take(t).to_pylist()
    an, bn, aad, bad = g("nosuf", ta), g("nosuf", tb), g("addr", ta), g("addr", tb)
    asq, bsq, anu, bnu = g("sq", ta), g("sq", tb), g("nums", ta), g("nums", tb)
    c = lambda x, y, sc: process.cpdist(x, y, scorer=sc, workers=1, dtype=np.float32)
    S = np.zeros((len(a), 5), np.float32)
    S[:, 0] = c(an, bn, fuzz.token_set_ratio) / 100
    S[:, 1] = c(an, bn, fuzz.ratio) / 100
    S[:, 2] = c(aad, bad, fuzz.token_set_ratio) / 100
    S[:, 4] = c(asq, bsq, JaroWinkler.normalized_similarity)
    for i in range(len(a)):
        x, y = set(anu[i].split()), set(bnu[i].split())
        if x and y:
            S[i, 3] = len(x & y) / len(x | y)
    return S


def record_sims(rec, a_idx, b_idx, workers, chunk=200_000):
    """[name tset, name ratio, addr tset, number jaccard, squashed-name JW] between records."""
    for k in ("nosuf", "sq", "addr", "nums"):
        _R[k] = rec[k]
    tasks = [(a_idx[s:s + chunk], b_idx[s:s + chunk]) for s in range(0, len(a_idx), chunk)]
    if workers <= 1 or len(tasks) <= 1:
        parts = [_sim_chunk(t) for t in tasks]
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            parts = list(pool.imap(_sim_chunk, tasks))
    return np.concatenate(parts) if parts else np.zeros((0, 5), np.float32)


def consensus_features(row, p1, b_idx, is_s3, rec, n_rows, workers, tau0=0.002, n_anchor=3):
    """-> (sel: pair indices that get a stage-2 score, F: float32 [len(sel), len(CONS_FEATS)])."""
    n = len(row)
    p1 = np.asarray(p1, np.float64)
    ok = b_idx >= 0
    pz = np.where(ok, p1, 0.0)
    # row landscape over ALL pairs
    o = np.lexsort((-pz, row))
    ro = row[o]
    cnt = np.bincount(row, minlength=n_rows)
    st = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    prank = np.empty(n, np.int64)
    prank[o] = np.arange(n) - st[ro]
    pmax = np.zeros(n_rows)
    psec = np.zeros(n_rows)
    first, second = o[prank[o] == 0], o[prank[o] == 1]
    pmax[row[first]] = pz[first]
    psec[row[second]] = pz[second]
    psum = np.bincount(row, weights=pz, minlength=n_rows)
    n50 = np.bincount(row, weights=(pz >= 0.5), minlength=n_rows)
    conf = pz >= 0.5
    n50_s3 = np.bincount(row, weights=conf & is_s3, minlength=n_rows)
    n50_s2 = n50 - n50_s3

    sel = np.flatnonzero((pz >= tau0) & ok)
    m = len(sel)
    F = np.zeros((m, len(CONS_FEATS)), np.float32)
    fi = {f: i for i, f in enumerate(CONS_FEATS)}
    if not m:
        return sel, F
    r = row[sel]
    ps = pz[sel]
    F[:, fi["p1"]] = ps
    F[:, fi["p1_logit"]] = np.log(np.clip(ps, 1e-6, 1 - 1e-6) / np.clip(1 - ps, 1e-6, 1))
    F[:, fi["p_rank"]] = prank[sel]
    F[:, fi["p_max_row"]] = pmax[r]
    F[:, fi["p_second_row"]] = psec[r]
    F[:, fi["p_sum_row"]] = psum[r]
    F[:, fi["n50_row"]] = n50[r]
    F[:, fi["p_gap_top"]] = pmax[r] - ps
    F[:, fi["p_ratio"]] = ps / np.maximum(pmax[r], 1e-9)
    F[:, fi["is_anchor0"]] = prank[sel] == 0
    s3 = is_s3[sel].astype(bool)
    self_conf = conf[sel]
    F[:, fi["n_conf_same_src"]] = np.where(s3, n50_s3[r], n50_s2[r]) - self_conf
    F[:, fi["n_conf_other_src"]] = np.where(s3, n50_s2[r], n50_s3[r])

    # anchors: the n_anchor highest-p pairs of each row (that are in sel)
    anc = -np.ones((n_rows, n_anchor), np.int64)
    top = o[(prank[o] < n_anchor)]
    top = top[(pz[top] >= tau0) & ok[top]]
    anc[row[top], prank[top]] = top
    # pair each selected candidate with every anchor of its row except itself
    cs, ks = [], []
    for j in range(n_anchor):
        aj = anc[r, j]
        v = (aj >= 0) & (aj != sel)
        cs.append(np.flatnonzero(v))
        ks.append(aj[v])
    ci = np.concatenate(cs)
    ai = np.concatenate(ks)
    slot = np.concatenate([np.full(len(c), j) for j, c in enumerate(cs)])
    S = record_sims(rec, b_idx[sel][ci], b_idx[ai], workers) if len(ci) else np.zeros((0, 5), np.float32)
    pa_ = pz[ai]
    # "o_*": similarity to the best anchor other than itself (lowest slot)
    best_slot = group_reduce(ci, slot, m, "min", 99).astype(np.int64)
    is_best = slot == best_slot[ci]
    bi = ci[is_best]
    for f, col in (("o_ntset", 0), ("o_nratio", 1), ("o_atset", 2), ("o_num", 3), ("o_jw", 4)):
        F[bi, fi[f]] = S[is_best, col]
    F[bi, fi["o_p"]] = pa_[is_best]
    # "sup_*": support from all other anchors, weighted by their probability
    for f, v in (("sup_name", pa_ * S[:, 0]), ("sup_addr", pa_ * S[:, 2]),
                 ("sup_both", pa_ * np.minimum(S[:, 0], S[:, 2]))):
        F[:, fi[f]] = group_reduce(ci, v, m, "max", 0.0)
    F[:, fi["sup_cnt"]] = np.bincount(ci, weights=pa_ * (S[:, 0] >= 0.85), minlength=m)
    return sel, F


def floor_p(p, tau0):
    """Canonical probability floor: anything below tau0/2 counts as exactly 0 in stage-2 row
    features and in every decision rule. The prefilter cascade only ever skips pairs below this
    floor, which makes predictions with and without the cascade identical."""
    p = np.asarray(p, np.float64)
    return np.where(p >= tau0 / 2, p, 0.0)


def stage2_matrix(X, sel, F2, feat_idx=None):
    Xs = X[sel] if feat_idx is None or len(feat_idx) == X.shape[1] else X[np.ix_(sel, feat_idx)]
    return np.hstack([Xs, F2]).astype(np.float32)


# =========================================================================== decision: expected F0.5
class EFDecider:
    """Per row, choose how many top candidates to predict by maximizing the expected F0.5
    under the model's probabilities (plug-in: E[F] ~ 1.25*sum_topk p / (0.25*(sum p + m_extra) + k)),
    vs predicting nothing (F=1 iff no true match: prod(1-p) * exp(-m_extra) * empty_bias).
    m_extra = expected true matches outside the candidates (blocking misses).
    The sort is done once; each (m_extra, empty_bias) call is a few vector ops."""

    def __init__(self, row, p, n_rows):
        n = len(p)
        self.n, self.n_rows = n, n_rows
        p = np.clip(np.asarray(p, np.float64), 0, 1)
        self.o = np.lexsort((-p, row))
        self.r = row[self.o]
        q = p[self.o]
        cnt = np.bincount(row, minlength=n_rows)
        self.st = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)
        self.nz = cnt > 0
        self.k = np.arange(n) - self.st[self.r] + 1
        cs = np.cumsum(q)
        self.cum = cs - np.concatenate([[0.0], cs])[self.st][self.r]
        self.tot = np.bincount(row, weights=p, minlength=n_rows)
        self.logq = np.bincount(row, weights=np.log(np.clip(1 - p, 1e-12, 1)), minlength=n_rows)

    def __call__(self, m_extra=0.0, empty_bias=1.0):
        pred = np.zeros(self.n, bool)
        if not self.n:
            return pred
        r, k, st, nz = self.r, self.k, self.st, self.nz
        E = 1.25 * self.cum / (0.25 * (self.tot[r] + m_extra) + k)
        bestE = np.full(self.n_rows, -1.0)
        bestE[nz] = np.maximum.reduceat(E, st[nz])
        kk = np.where(E >= bestE[r] - 1e-12, k, np.iinfo(np.int64).max)
        kstar = np.zeros(self.n_rows, np.int64)
        kstar[nz] = np.minimum.reduceat(kk, st[nz])
        E0 = np.exp(self.logq - m_extra) * empty_bias
        pred[self.o] = (bestE > E0)[r] & (k <= kstar[r])
        return pred


def ef_decide(row, p, n_rows, m_extra=0.0, empty_bias=1.0):
    return EFDecider(row, p, n_rows)(m_extra, empty_bias)


def owner_exclusive(pred, p, key):
    o = np.lexsort((-p, key))
    po = pred[o]
    idx, k = o[po], key[o][po]
    first = np.ones(len(idx), bool)
    first[1:] = k[1:] != k[:-1]
    out = np.zeros(len(pred), bool)
    out[idx[first]] = True
    return out


def apply_rule(rule, row, p, key, n_rows, oe_local=None):
    """rule: {"kind": "thresh", t_top, t_extra} or {"kind": "ef", m_extra, empty_bias}; + owner_exclusive."""
    if rule.get("kind", "thresh") == "ef":
        pred = ef_decide(row, p, n_rows, rule["m_extra"], rule["empty_bias"])
    else:
        pred = Decider(row, p, key, n_rows)(rule["t_top"], rule["t_extra"], False)
    oe = rule["owner_exclusive"] if oe_local is None else oe_local
    return owner_exclusive(pred, p, key) if oe else pred

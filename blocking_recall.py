"""
Blocking (candidate generation) for the Business Entity Resolution challenge -- v7.

Same blocking keys as v6 (squash, rare_name, rare_ngram, addr_num, addr_pin,
rare_addr, addr_pair, name_pair, optional TF-IDF), rebuilt for scale, after a
real US run (1.32M S1 / 3.0M S2 / 3.17M S3) measured:

  * MEMORY: normalizing S2 cost +10.4GB and S3 +11.7GB, about 3.5KB per row.
    That came from per-row Python objects (frozensets, lists of n-gram strings,
    several tiny numpy arrays per row, each ~100B of header). The indexes and
    the fork itself were small by comparison.
    -> v7 keeps every per-row key list as ONE flat int64 array + offsets (CSR)
       per column. There are no per-row Python objects after normalization, and
       one country is processed and freed before the next is loaded.

  * SPEED: phase-1 lookup ran at ~127 S1 rows/s per worker, since each S1 row had a
    ~2,000-candidate union scored in a Python loop (Counter + per-candidate
    jaccard).
    -> v7 ranks each row with a handful of vectorized numpy calls (np.unique
       for agreement counts, a gather + searchsorted for both jaccards,
       lexsort for the order).

  * DIAGNOSIS: recall@50 alone can't tell "the keys never found the true match"
    apart from "the true match was found but ranked below top-k". v7 records,
    on train, the RANK of every true match inside the full candidate union, and
    prints recall@k for k = 10 ... 1000 plus the not-in-union share and the
    F0.5 ceiling implied at your --topk.

  * OPTIONAL LEARNED RANKER (--fit-ranker N): agreement-count-then-jaccard is
    a fixed lexicographic rule (for example, an exact squashed-name hit counts
    the same as a shared house number). With --fit-ranker, v7 collects
    (family-hit indicators, name/addr jaccard) for N train S1 rows, fits a
    logistic regression, ranks with it, and prints the baseline and learned
    recall@k curves side by side on held-out rows. The weights are saved
    (--save-ranker) and reused on test (--ranker-weights). Training data is the
    challenge's own train ground truth only.

PYTHONHASHSEED is forced to 0 (the script re-execs itself), so hashed keys are
identical across runs and machines: a --countries split concatenates to exactly
the output of an unsplit run.

Output parquet columns (compatible with train_matcher.py / audit_misses.py):
  source1_entity_id, country, candidate_entity_ids, candidate_agreement,
  candidate_coarse_score, candidate_rank_score, n_before_prune, tfidf_ran,
  hits_<family> (union size contributed per key family), true_ranks (train only)
"""
import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import gc
import json
import multiprocessing as mp
import re
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from normalize import clean_name, clean_address
from build_translit_dict import tokens as split_tokens, is_native

_SQUASH_RE = re.compile(r"[^a-z0-9]")
COLS = ("squash", "name_tok", "ngram", "addr_tok", "addr_num", "addr_pin", "pair_addr", "pair_name")
# (family name, S1/corpus column it is built from). rare_* families index only
# each corpus row's k rarest keys; the query side always probes with ALL of its keys.
BASE_FAMS = [("squash", "squash"), ("rare_name", "name_tok"), ("rare_ngram", "ngram"),
             ("addr_num", "addr_num"), ("addr_pin", "addr_pin"), ("rare_addr", "addr_tok")]
PAIR_FAMS = [("addr_pair", "pair_addr"), ("name_pair", "pair_name")]
RECALL_KS = (10, 25, 50, 100, 200, 300, 500, 1000)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_mem(stage):
    """System-wide memory from /proc/meminfo. Summed ps RSS double-counts pages shared across forks."""
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                p = line.split()
                if len(p) >= 2 and p[1].isdigit():
                    info[p[0].rstrip(":")] = int(p[1])
        tot, av = info["MemTotal"] / 1024 ** 2, info["MemAvailable"] / 1024 ** 2
        st, sf = info.get("SwapTotal", 0) / 1024 ** 2, info.get("SwapFree", 0) / 1024 ** 2
        log(f"  [mem] {stage}: used={tot - av:.2f}GB avail={av:.2f}GB / {tot:.2f}GB | "
            f"swap used={st - sf:.2f}GB / {st:.2f}GB")
    except Exception as e:  # non-Linux
        log(f"  [mem] {stage}: n/a ({e})")


# --------------------------------------------------------------------------- I/O
_TSV_COLS = ("entity_id", "business_name", "business_address", "country")


def read_table(path, countries=None) -> pa.Table:
    """Arrow table (compact strings, no per-cell Python objects), filtered to
    `countries` if given. Falls back to pandas if pyarrow's parser rejects the file."""
    try:
        with open(path, encoding="utf-8") as f:
            header = f.readline().rstrip("\n").rstrip("\r").split("\t")
        tbl = pacsv.read_csv(
            path,
            read_options=pacsv.ReadOptions(block_size=64 << 20),
            parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
            convert_options=pacsv.ConvertOptions(
                column_types={c: pa.string() for c in header},
                strings_can_be_null=False, quoted_strings_can_be_null=False),
        )
    except Exception as e:
        log(f"  pyarrow CSV parse failed ({e}); falling back to pandas for {path}")
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False, quoting=3)
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        del df
    if countries:
        tbl = tbl.filter(pc.is_in(tbl["country"], value_set=pa.array(sorted(countries))))
    return tbl.combine_chunks()


# --------------------------------------------------------------------------- normalization -> CSR
_ADDR_STOP = frozenset("""
street st road rd avenue ave av boulevard blvd drive dr lane ln highway hwy way
suite ste unit apt apartment floor fl flat plot no number block blk sector near
opp opposite building bldg house room shop the of and at po ps post office saint
null none na nan
""".split())
_NAME_STOP = frozenset("""
inc llc ltd pvt private limited corp corporation co company lp llp plc pc pa
pllc the and of ms sarl sas sa gmbh oyj ab bv nv
""".split())


def latinize_ci(text, translit_dict):
    if not is_native(text):
        return text
    return " ".join(translit_dict.get(raw.casefold().strip(".,()[]{}'\""), raw) for raw in text.split())


def char_ngrams(s, n=4):
    return [s[i:i + n] for i in range(len(s) - n + 1)] if len(s) >= n else ([s] if s else [])


def compound_keys(name_tokens, addr_nums, addr_words):
    """(house-number x address-word) and (name-token x name-token) keys -- identical to v6."""
    nums = []
    for n in addr_nums:
        n = n.lstrip("0") or "0"
        if len(n) <= 6 and n not in nums:
            nums.append(n)
        if len(nums) >= 3:
            break
    words = sorted(w for w in addr_words if w not in _ADDR_STOP)[:10]
    a = {hash(("a", n, w)) for n in nums for w in words}
    toks = sorted(t for t in name_tokens if t not in _NAME_STOP and len(t) >= 2)[:6]
    nm = {hash(("n", toks[i], toks[j])) for i in range(len(toks)) for j in range(i + 1, len(toks))}
    return a, nm


_NORM = {}  # set in the parent before forking normalize workers


def _norm_chunk(args):
    names, addrs = args
    td, want_sq = _NORM["translit"], _NORM["want_squashed"]
    flat = {c: [] for c in COLS}
    lens = {c: [] for c in COLS}
    sq_out = []
    for raw_name, raw_addr in zip(names, addrs):
        ni = clean_name(latinize_ci(raw_name, td))
        ai = clean_address(raw_addr)
        v0 = ni["variants"][0]
        sq = v0["squashed"]
        sqa = ni["variants"][-1]["squashed"] if len(ni["variants"]) > 1 else ""
        nosuf = v0["no_suffix"]
        name_tokens = set(split_tokens(nosuf))
        words = ai["word_tokens"]
        nums = ai["numeric_tokens"]
        pa_keys, pn_keys = compound_keys(name_tokens, nums, words)
        keys = {
            "squash": {hash(s) for s in (sq, sqa) if s},
            "name_tok": {hash(t) for t in name_tokens},
            "ngram": {hash(g) for g in char_ngrams(_SQUASH_RE.sub("", nosuf), 4)},
            "addr_tok": {hash(t) for t in words},
            "addr_num": {hash(n[-4:]) for n in nums if len(n) >= 3},
            "addr_pin": {hash(n) for n in nums if len(n) >= 5},
            "pair_addr": pa_keys,
            "pair_name": pn_keys,
        }
        for c in COLS:
            s = sorted(keys[c])            # sorted + unique: required by jaccard/searchsorted
            flat[c].extend(s)
            lens[c].append(len(s))
        if want_sq:
            sq_out.append(sq)
    return ({c: (np.array(flat[c], dtype=np.int64), np.array(lens[c], dtype=np.int64)) for c in COLS},
            sq_out)


def normalize_table(tbl: pa.Table, translit, workers, want_squashed, chunk=50_000):
    """-> {"n", "cols": {col: (flat int64, offsets int64)}, "squashed": object array or None}"""
    n = tbl.num_rows
    _NORM["translit"], _NORM["want_squashed"] = translit, want_squashed
    names, addrs = tbl["business_name"], tbl["business_address"]

    def tasks():
        for s in range(0, n, chunk):
            yield names.slice(s, chunk).to_pylist(), addrs.slice(s, chunk).to_pylist()

    if workers <= 1 or n < 2 * chunk:
        parts = [_norm_chunk(t) for t in tasks()]
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            parts = list(pool.imap(_norm_chunk, tasks()))
    cols = {}
    for c in COLS:
        fl = np.concatenate([p[0][c][0] for p in parts]) if parts else np.zeros(0, np.int64)
        ln = np.concatenate([p[0][c][1] for p in parts]) if parts else np.zeros(0, np.int64)
        off = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(ln, out=off[1:])
        cols[c] = (fl, off)
        for p in parts:              # free each chunk's piece as soon as it's merged
            p[0][c] = None
    sq = np.array([s for p in parts for s in p[1]], dtype=object) if want_squashed else None
    del parts
    return {"n": n, "cols": cols, "squashed": sq}


# --------------------------------------------------------------------------- index building
def _row_ids(off):
    return np.repeat(np.arange(len(off) - 1, dtype=np.int32), np.diff(off))


def rarest_k_csr(flat, off, k):
    """Per row, keep the k keys with the lowest document frequency in this corpus
    (ties broken by key value, deterministic). Rows are already de-duplicated."""
    if len(flat) == 0:
        return flat, off
    rows = _row_ids(off)
    _, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
    freq = cnt[inv]
    order = np.lexsort((flat, freq, rows))
    rows_s = rows[order]
    first = off[:-1][rows_s]                     # first slot of each row (rows sorted -> same layout)
    rank = np.arange(len(rows_s)) - first
    keep = order[rank < k]
    keep.sort()                                  # back to row-major order
    f2 = flat[keep]
    ln = np.bincount(rows[keep], minlength=len(off) - 1)
    off2 = np.zeros(len(off), dtype=np.int64)
    np.cumsum(ln, out=off2[1:])
    return f2, off2


def capped_postings(flat, off, cap, base):
    """(key, global position) postings with buckets larger than `cap` removed."""
    if len(flat) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int32), 0, 0
    pos = _row_ids(off) + np.int32(base)
    order = np.argsort(flat, kind="stable")
    k, p = flat[order], pos[order]
    del order
    _, cnt = np.unique(k, return_counts=True)
    good = cnt <= cap
    keep = np.repeat(good, cnt)
    return k[keep], p[keep], int((~good).sum()), int(good.sum())


def merge_index(parts):
    """Merge per-source capped postings into one sorted index: keys/starts/counts/pos."""
    k = np.concatenate([x[0] for x in parts])
    p = np.concatenate([x[1] for x in parts])
    if len(k) == 0:
        e = np.zeros(0, np.int64)
        return {"keys": e, "starts": e, "counts": e, "pos": np.zeros(0, np.int32)}
    order = np.argsort(k, kind="stable")
    k, p = k[order], p[order]
    keys, starts, counts = np.unique(k, return_index=True, return_counts=True)
    return {"keys": keys, "starts": starts.astype(np.int64), "counts": counts.astype(np.int64), "pos": p}


def concat_csr(a, b):
    fa, oa = a
    fb, ob = b
    return np.concatenate([fa, fb]), np.concatenate([oa, ob[1:] + oa[-1]])


# --------------------------------------------------------------------------- per-row ranking (workers)
_SHARED = {}


def _gather(flat, off, idx):
    """Concatenate CSR rows `idx`; returns (values, lengths)."""
    st = off[idx]
    ln = off[idx + 1] - st
    tot = int(ln.sum())
    if tot == 0:
        return flat[:0], ln
    g = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(ln) - ln, ln) + np.repeat(st, ln)
    return flat[g], ln


def lookup(q, ix):
    keys = ix["keys"]
    if len(q) == 0 or len(keys) == 0:
        return None
    i = np.searchsorted(keys, q)
    np.minimum(i, len(keys) - 1, out=i)
    j = i[keys[i] == q]
    if len(j) == 0:
        return None
    if len(j) == 1:
        s = ix["starts"][j[0]]
        return ix["pos"][s:s + ix["counts"][j[0]]]
    st, ct = ix["starts"][j], ix["counts"][j]
    tot = int(ct.sum())
    g = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(ct) - ct, ct) + np.repeat(st, ct)
    return np.unique(ix["pos"][g])


def jaccard_many(q, flat, off, cand):
    """Jaccard of sorted-unique query keys q against every candidate's key row."""
    toks, ln = _gather(flat, off, cand)
    lq = len(q)
    if lq == 0 or len(toks) == 0:
        return np.zeros(len(cand))
    s = np.searchsorted(q, toks)
    np.minimum(s, lq - 1, out=s)
    hit = (q[s] == toks)
    inter = np.bincount(np.repeat(np.arange(len(cand)), ln), weights=hit, minlength=len(cand))
    union = lq + ln - inter
    return np.divide(inter, union, out=np.zeros(len(cand)), where=union > 0)


def n_features(n_fam):
    return n_fam + 1 + 4       # family indicators + tfidf indicator + jn, ja, jn*ja, is_s3


def feature_matrix(hits, cand, jn, ja, n_fam, n2):
    X = np.zeros((len(cand), n_features(n_fam)), dtype=np.float64)
    for f, h in hits:
        X[np.searchsorted(cand, h), f] = 1.0
    X[:, n_fam + 1] = jn
    X[:, n_fam + 2] = ja
    X[:, n_fam + 3] = jn * ja
    X[:, n_fam + 4] = cand >= n2
    return X


def _rank_worker(rows):
    S = _SHARED
    Q, idx, fams, topk = S["qcols"], S["idx"], S["fams"], S["topk"]
    nf = len(fams)
    cnf, cno = S["cname"]
    caf, cao = S["caddr"]
    n2, W, gt, tf, mode = S["n2"], S["weights"], S["gt"], S["tfidf"], S["mode"]
    qnf, qno = Q["name_tok"]
    qaf, qao = Q["addr_tok"]
    m = len(rows)
    rng = np.random.default_rng(int(rows[0]) if m else 0)
    if mode == "rank":
        o_c = np.full((m, topk), -1, np.int32)
        o_a = np.zeros((m, topk), np.int8)
        o_s = np.zeros((m, topk), np.float32)
        o_r = np.zeros((m, topk), np.float32)
        o_n = np.zeros(m, np.int32)
        o_h = np.zeros((m, nf + 1), np.int32)
        o_m = np.zeros((m, topk), np.int16)
        o_jn = np.zeros((m, topk), np.float32)
        o_ja = np.zeros((m, topk), np.float32)
    tr_model, tr_base = [], []
    Xs, ys = [], []
    empty = np.zeros(0, np.int32)
    for i, r in enumerate(rows):
        hits = []
        for f, (fam, col) in enumerate(fams):
            fl, of = Q[col]
            h = lookup(fl[of[r]:of[r + 1]], idx[fam])
            if h is not None:
                hits.append((f, h))
        if tf is not None:
            th = tf[0][tf[1][r]:tf[1][r + 1]]
            if len(th):
                hits.append((nf, th))
        t = gt[0][gt[1][r]:gt[1][r + 1]] if gt is not None else empty
        if not hits:
            if gt is not None:
                tr_model.append(np.full(len(t), -1, np.int32))
                tr_base.append(np.full(len(t), -1, np.int32))
            continue
        allp = np.concatenate([h for _, h in hits]) if len(hits) > 1 else hits[0][1]
        cand, agree = np.unique(allp, return_counts=True)
        n = len(cand)
        jn = jaccard_many(qnf[qno[r]:qno[r + 1]], cnf, cno, cand)
        ja = jaccard_many(qaf[qao[r]:qao[r + 1]], caf, cao, cand)
        coarse = 0.5 * jn + 0.5 * ja
        base_order = np.lexsort((cand, -coarse, -agree))
        X = None
        if W is not None or mode == "collect":
            X = feature_matrix(hits, cand, jn, ja, nf, n2)
        if W is not None:
            rs = X @ W
            order = np.lexsort((cand, -rs))
        else:
            rs, order = coarse, base_order
        R_ = S.get("reserve", 0)
        if mode == "rank" and R_ and n > topk:
            # the last R_ of the top-k slots go to exact-name candidates (squashed-name key hit or the
            # same name-token set) that the ranker pushed below the cutoff -- e.g. blank-address copies
            # of a common name, which score low on every address signal
            is_ex = jn >= 0.999
            for f, h in hits:
                if f == S["squash_f"]:
                    is_ex[np.searchsorted(cand, h)] = True
            tail = order[topk - R_:]
            te = is_ex[tail]
            take = te & (np.cumsum(te) <= R_)
            if take.any():
                order = np.concatenate([order[:topk - R_], tail[take], tail[~take]])

        if mode == "collect":
            pos = np.zeros(n, bool)
            if len(t):
                j = np.minimum(np.searchsorted(cand, t), n - 1)
                pos[j[cand[j] == t]] = True
            neg = base_order[~pos[base_order]]
            hard = neg[:100]
            rest = neg[100:]
            rnd = rng.choice(rest, size=min(50, len(rest)), replace=False) if len(rest) else rest
            sel = np.concatenate([np.flatnonzero(pos), hard, rnd])
            Xs.append(X[sel].astype(np.float32))
            ys.append(pos[sel])
            continue

        k = min(topk, n)
        sel = order[:k]
        o_c[i, :k] = cand[sel]
        o_a[i, :k] = np.minimum(agree[sel], 127)
        o_s[i, :k] = coarse[sel]
        o_r[i, :k] = rs[sel]
        o_n[i] = n
        ind = np.zeros(n, np.int16)
        for f, h in hits:
            o_h[i, f] = len(h)
            ind[np.searchsorted(cand, h)] |= np.int16(1 << f)
        o_m[i, :k] = ind[sel]
        o_jn[i, :k] = jn[sel]
        o_ja[i, :k] = ja[sel]
        if gt is not None:
            if len(t):
                j = np.minimum(np.searchsorted(cand, t), n - 1)
                present = cand[j] == t
                rk = np.empty(n, np.int32)
                rk[order] = np.arange(n, dtype=np.int32)
                tr_model.append(np.where(present, rk[j], -1).astype(np.int32))
                if W is not None:
                    rk[base_order] = np.arange(n, dtype=np.int32)
                tr_base.append(np.where(present, rk[j], -1).astype(np.int32))
            else:
                tr_model.append(empty)
                tr_base.append(empty)
    if mode == "collect":
        if not Xs:
            return np.zeros((0, n_features(nf)), np.float32), np.zeros(0, bool)
        return np.concatenate(Xs), np.concatenate(ys)
    return rows, o_c, o_a, o_s, o_r, o_n, o_h, o_m, o_jn, o_ja, tr_model, tr_base


def run_pool(fn, row_chunks, workers):
    if workers <= 1 or len(row_chunks) <= 1:
        for ch in row_chunks:
            yield fn(ch)
        return
    with mp.get_context("fork").Pool(workers) as pool:
        for res in pool.imap_unordered(fn, row_chunks):
            yield res


# --------------------------------------------------------------------------- TF-IDF (optional)
def tfidf_hits_csr(queries, corpus, k, max_df, n_threads, batch=5000):
    """Char-3/4-gram TF-IDF top-k over corpus = S2 then S3 (so column index = global position)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sparse_dot_topn import sp_matmul_topn
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), dtype=np.float32, max_df=max_df)
    cm = vec.fit_transform(corpus)
    cmT = cm.T.tocsr()
    k_eff = min(k, cm.shape[0])
    flats, lens = [], []
    for s in range(0, len(queries), batch):
        res = sp_matmul_topn(vec.transform(queries[s:s + batch]), cmT, top_n=k_eff,
                             threshold=0.0, n_threads=n_threads).tocsr()
        flats.append(res.indices.astype(np.int32))
        lens.append(np.diff(res.indptr))
    del vec, cm, cmT
    gc.collect()
    ln = np.concatenate(lens) if lens else np.zeros(0, np.int64)
    off = np.zeros(len(ln) + 1, np.int64)
    np.cumsum(ln, out=off[1:])
    return (np.concatenate(flats) if flats else np.zeros(0, np.int32)), off


# --------------------------------------------------------------------------- reporting
# ------------------------------------------------------------------ reverse (competing-S1) statistics
_NP_FAST_AT = tuple(int(x) for x in np.__version__.split(".")[:2]) >= (1, 25)


def group_max(keys, vals, n):
    """max of vals per key (keys in [0, n)); -inf where a key has no values."""
    if _NP_FAST_AT:
        out = np.full(n, -np.inf, np.float32)
        np.maximum.at(out, keys, vals)
        return out
    import pandas as pd       # numpy < 1.25: ufunc.at is ~100x slower
    s = pd.Series(vals).groupby(keys).max()
    out = np.full(n, -np.inf, np.float32)
    out[s.index.to_numpy()] = s.to_numpy()
    return out


def reverse_stats(res, n_corpus):
    """For every (S1 row, candidate record) pair: how well do the OTHER S1 rows that also list this
    record match it? Returns 1-D arrays in the order of res['c'][res['c'] >= 0]:
      rev_n  = number of other S1 rows listing the record
      rev_r / rev_jn / rev_ja = best rank_score / name jaccard / address jaccard among those other rows
                                (-1 when no other S1 lists it)
    Only meaningful when EVERY S1 row of the country was ranked (--sample-s1 0)."""
    c = res["c"]
    valid = c >= 0
    rec = c[valid]
    row = np.repeat(np.arange(len(c), dtype=np.int32), valid.sum(1))
    cnt = np.bincount(rec, minlength=n_corpus)
    out = {"rev_n": (cnt[rec] - 1).astype(np.int32)}
    for name, key in (("rev_r", "r"), ("rev_jn", "jn"), ("rev_ja", "ja")):
        v = res[key][valid].astype(np.float32)
        best = group_max(rec, v, n_corpus)
        isb = v == best[rec]
        bestrow = np.full(n_corpus, -1, np.int32)
        bestrow[rec[isb]] = row[isb]
        nb = np.bincount(rec[isb], minlength=n_corpus)
        second = group_max(rec[~isb], v[~isb], n_corpus)
        second = np.where(nb >= 2, best, second)
        other = np.where(row == bestrow[rec], second[rec], best[rec])
        out[name] = np.where(np.isfinite(other), other, -1.0).astype(np.float32)
        del v, best, isb, bestrow, nb, second, other
    return out


def f_half_ceiling(found, n_true):
    """Macro F0.5 with a perfect matcher on these candidates: singletons 1.0,
    others 1.25R/(0.25+R) with precision 1 (0 if nothing found)."""
    R = np.divide(found, n_true, out=np.zeros(len(n_true)), where=n_true > 0)
    F = np.where(n_true == 0, 1.0, np.divide(1.25 * R, 0.25 + R, out=np.zeros(len(R)), where=R > 0))
    return F.mean()


def recall_report(ranks_list, label, topk):
    n_true = np.array([len(x) for x in ranks_list], dtype=np.int64)
    if n_true.sum() == 0:
        return
    flat = np.concatenate([x for x in ranks_list if len(x)])
    row = np.repeat(np.arange(len(ranks_list)), n_true)
    has = n_true > 0
    lines = [f"=== {label}: {has.sum():,} S1 rows with >=1 true match, {n_true.sum():,} true pairs ==="]
    lines.append(f"{'k':>8} {'macro recall':>13} {'micro recall':>13} {'S1 fully found':>15}")
    for k in list(RECALL_KS) + ["union"]:
        ok = (flat >= 0) if k == "union" else ((flat >= 0) & (flat < k))
        found = np.bincount(row, weights=ok, minlength=len(ranks_list))
        mac = (found[has] / n_true[has]).mean() * 100
        mic = found.sum() / n_true.sum() * 100
        full = (found[has] == n_true[has]).mean() * 100
        tag = "  <- --topk" if k == topk else ""
        lines.append(f"{str(k):>8} {mac:12.2f}% {mic:12.2f}% {full:14.2f}%{tag}")
    ok = (flat >= 0) & (flat < topk)
    found = np.bincount(row, weights=ok, minlength=len(ranks_list))
    lines.append(f"true pairs NOT in the candidate union at all (keys missed them): "
                 f"{(flat < 0).mean() * 100:.2f}%  | in union but ranked >= {topk}: "
                 f"{((flat >= topk)).mean() * 100:.2f}%")
    lines.append(f"F0.5 ceiling at --topk {topk} with a perfect matcher (singletons included): "
                 f"{f_half_ceiling(found, n_true):.4f}")
    for ln in lines:
        log(ln)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--translit-dict", default="translit_dict.json")
    ap.add_argument("--sample-s1", type=int, default=0, help="0 = all S1 rows")
    ap.add_argument("--reserve-exact", type=int, default=0,
                    help="reserve up to this many of the top-k slots for exact-name candidates ranked "
                         "below the cutoff (use the SAME value for train and test candidates)")
    ap.add_argument("--write-sample", type=int, default=0,
                    help="rank ALL S1 rows (so the reverse competing-S1 columns are exact) but write only "
                         "this many random rows per country (training parquets); 0 = write all")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--countries", default=None, help="comma-separated labels; default all")
    ap.add_argument("--rare-k", type=int, default=3)
    ap.add_argument("--rare-k-ngram", type=int, default=2)
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--pair-keys", type=int, default=1)
    ap.add_argument("--pair-cap", type=int, default=300)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--tfidf-k", type=int, default=0)
    ap.add_argument("--tfidf-weak-threshold", type=int, default=10)
    ap.add_argument("--tfidf-min-confident-agreement", type=int, default=2)
    ap.add_argument("--tfidf-min-confident-score", type=float, default=0.15)
    ap.add_argument("--tfidf-max-df", type=float, default=0.3)
    ap.add_argument("--tfidf-threads", type=int, default=4)
    ap.add_argument("--fit-ranker", type=int, default=0,
                    help="train only: fit a logistic first-stage ranker on this many S1 rows per country "
                         "(held out from the recall report), then rank everything with it")
    ap.add_argument("--save-ranker", default=None,
                    help="default ranker_<countries>.json, so per-country runs never overwrite each other")
    ap.add_argument("--ranker-weights", default=None,
                    help="comma-separated weight json(s) from --save-ranker, for test runs")
    ap.add_argument("--norm-workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--lookup-workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--chunk-rows", type=int, default=2000)
    ap.add_argument("--out", default="candidates.parquet")
    a = ap.parse_args()
    t_start = time.time()
    log(f"v7 | PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')} | norm-workers={a.norm_workers} "
        f"lookup-workers={a.lookup_workers} topk={a.topk}")

    with open(a.translit_dict, encoding="utf-8") as f:
        translit = json.load(f)
    wanted = {c.strip() for c in a.countries.split(",") if c.strip()} if a.countries else None
    d = os.path.join(a.data_dir, a.split)
    s1 = read_table(os.path.join(d, f"{a.split}_source1.tsv"), wanted)
    s2 = read_table(os.path.join(d, f"{a.split}_source2.tsv"), wanted)
    s3 = read_table(os.path.join(d, f"{a.split}_source3.tsv"), wanted)
    log(f"loaded S1={s1.num_rows:,} S2={s2.num_rows:,} S3={s3.num_rows:,}"
        + (f" (countries {sorted(wanted)})" if wanted else ""))
    if s1.num_rows == 0:
        raise SystemExit("zero S1 rows after --countries filter -- check the label spelling")
    if a.sample_s1 and a.sample_s1 < s1.num_rows:
        pick = np.random.RandomState(a.seed).choice(s1.num_rows, a.sample_s1, replace=False)
        s1 = s1.take(pa.array(pick))
        log(f"sampled S1 down to {s1.num_rows:,}")
    log_mem("after load")

    gt_df = None
    gt_path = os.path.join(a.data_dir, "train", "train_ground_truth.tsv")
    if a.split == "train" and os.path.exists(gt_path):
        gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False, na_filter=False, quoting=3)
        gt_df = gt_df[gt_df["source1_entity_id"].isin(set(s1["entity_id"].to_pylist()))]
        gt_df = gt_df.set_index("source1_entity_id")["matched_entity_ids"]

    loaded_w = {}
    if a.ranker_weights:
        for p in a.ranker_weights.split(","):
            with open(p) as f:
                loaded_w.update(json.load(f))
        log(f"loaded ranker weights for: {sorted(loaded_w)}")
    fitted_w = {}

    fams = BASE_FAMS + (PAIR_FAMS if a.pair_keys else [])
    fam_names = [f for f, _ in fams] + ["tfidf"]
    log("family bits for candidate_family_mask: " + ", ".join(f"bit{i}={n}" for i, n in enumerate(fam_names)))
    if a.save_ranker is None:
        a.save_ranker = f"ranker_{'_'.join(sorted(wanted)) if wanted else 'all'}.json"
    if os.path.exists(a.out):
        log(f"WARNING: --out {a.out} exists and will be overwritten")
    writer = None
    all_tr_model, all_tr_base, all_heldout = [], [], []

    for country in sorted(set(s1["country"].to_pylist())):
        tc = time.time()
        mask = lambda t: pc.equal(t["country"], country)
        q_tbl, c2_tbl, c3_tbl = s1.filter(mask(s1)), s2.filter(mask(s2)), s3.filter(mask(s3))
        log(f"country={country}: S1={q_tbl.num_rows:,} S2={c2_tbl.num_rows:,} S3={c3_tbl.num_rows:,} -- normalizing ...")
        want_sq = a.tfidf_k > 0
        Q = normalize_table(q_tbl, translit, a.norm_workers, want_sq)
        C2 = normalize_table(c2_tbl, translit, a.norm_workers, want_sq)
        C3 = normalize_table(c3_tbl, translit, a.norm_workers, want_sq)
        n2 = C2["n"]
        log_mem(f"country={country}: after normalize ({time.time() - tc:.0f}s)")

        idx, stats = {}, []
        for fam, col in fams:
            cap = a.pair_cap if fam in ("addr_pair", "name_pair") else a.cap
            k = {"rare_name": a.rare_k, "rare_ngram": a.rare_k_ngram, "rare_addr": a.rare_k}.get(fam)
            parts = []
            for C, base in ((C2, 0), (C3, n2)):
                fl, of = C["cols"][col]
                if k is not None:
                    fl, of = rarest_k_csr(fl, of, k)
                kk, pp, dropped, kept = capped_postings(fl, of, cap, base)
                parts.append((kk, pp))
                stats.append(f"{fam}:{kept:,}/{dropped:,}")
            idx[fam] = merge_index(parts)
            del parts
        log(f"country={country}: index buckets kept/dropped-over-cap (s2 then s3): {' '.join(stats)}")
        cname = concat_csr(C2["cols"]["name_tok"], C3["cols"]["name_tok"])
        caddr = concat_csr(C2["cols"]["addr_tok"], C3["cols"]["addr_tok"])
        corpus_sq = np.concatenate([C2["squashed"], C3["squashed"]]) if want_sq else None
        del C2, C3
        gc.collect()
        log_mem(f"country={country}: after index build")

        corpus_ids = pa.concat_arrays([c2_tbl["entity_id"].combine_chunks(),
                                       c3_tbl["entity_id"].combine_chunks()])
        q_ids = q_tbl["entity_id"].combine_chunks()
        nq = q_tbl.num_rows

        gt = None
        if gt_df is not None:
            lists = gt_df.reindex(q_ids.to_pylist()).fillna("").tolist()
            split = [[x.strip() for x in s.split(",") if x.strip()] for s in lists]
            ln = np.array([len(x) for x in split], dtype=np.int64)
            flat_ids = pa.array([x for xs in split for x in xs], type=pa.string())
            posn = pc.index_in(flat_ids, value_set=corpus_ids).to_numpy(zero_copy_only=False)
            posn = np.where(np.isnan(posn.astype(float)), -2, posn).astype(np.int32)
            off = np.zeros(nq + 1, np.int64)
            np.cumsum(ln, out=off[1:])
            gt = (posn, off)
            miss = int((posn == -2).sum())
            if miss:
                log(f"  note: {miss:,} true ids not present in this country's S2/S3 (counted as misses)")

        _SHARED.update(qcols=Q["cols"], idx=idx, fams=fams, topk=a.topk, cname=cname, caddr=caddr,
                       n2=n2, weights=None, gt=gt, tfidf=None, mode="rank", reserve=a.reserve_exact,
                       squash_f=[f for f, _ in fams].index("squash"))

        all_rows = np.arange(nq, dtype=np.int64)
        heldout = np.ones(nq, bool)
        if a.fit_ranker and gt is not None:
            fit_rows = np.sort(np.random.RandomState(a.seed + 7).choice(nq, min(a.fit_ranker, nq), replace=False))
            heldout[fit_rows] = False
            _SHARED["mode"] = "collect"
            chunks = [fit_rows[i:i + a.chunk_rows] for i in range(0, len(fit_rows), a.chunk_rows)]
            Xs, ys = zip(*run_pool(_rank_worker, chunks, a.lookup_workers))
            X, y = np.concatenate(Xs), np.concatenate(ys)
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
            lr.fit(X.astype(np.float64), y)
            w = lr.coef_[0].copy()
            # TF-IDF never runs during collection (phase 1 only) -> give it rare_ngram's weight.
            w[len(fams)] = w[[f for f, _ in fams].index("rare_ngram")]
            fitted_w[country] = {"weights": w.tolist(), "families": fam_names,
                                 "features": fam_names + ["jn", "ja", "jn_x_ja", "is_s3"]}
            log(f"country={country}: fitted ranker on {len(fit_rows):,} rows / {len(y):,} candidate pairs "
                f"({y.mean() * 100:.1f}% positive): " + ", ".join(
                    f"{n}={v:+.2f}" for n, v in zip(fitted_w[country]["features"], w)))
            _SHARED["weights"] = w
            _SHARED["mode"] = "rank"
        elif loaded_w:
            wd = loaded_w.get(country)
            if wd is None:  # unseen country (e.g. France): average the known countries' weights
                wd = {"weights": np.mean([v["weights"] for v in loaded_w.values()], axis=0).tolist()}
                log(f"country={country}: no fitted ranker -> using the mean of {sorted(loaded_w)}")
            _SHARED["weights"] = np.array(wd["weights"])

        res = {"c": np.full((nq, a.topk), -1, np.int32), "a": np.zeros((nq, a.topk), np.int8),
               "s": np.zeros((nq, a.topk), np.float32), "r": np.zeros((nq, a.topk), np.float32),
               "n": np.zeros(nq, np.int32), "h": np.zeros((nq, len(fams) + 1), np.int32),
               "tfidf_ran": np.zeros(nq, bool), "m": np.zeros((nq, a.topk), np.int16),
               "jn": np.zeros((nq, a.topk), np.float32), "ja": np.zeros((nq, a.topk), np.float32)}
        trm, trb = [None] * nq, [None] * nq

        def absorb(out):
            rows, oc, oa, os_, or_, on, oh, om, ojn, oja, tm, tb = out
            res["m"][rows], res["jn"][rows], res["ja"][rows] = om, ojn, oja
            res["c"][rows], res["a"][rows], res["s"][rows], res["r"][rows] = oc, oa, os_, or_
            res["n"][rows], res["h"][rows] = on, oh
            if gt is not None:
                for rr, x, y2 in zip(rows, tm, tb):
                    trm[rr], trb[rr] = x, y2

        log(f"country={country}: ranking {nq:,} S1 rows across {a.lookup_workers} workers ...")
        log_mem(f"country={country}: before lookup fork")
        tl = time.time()
        chunks = [all_rows[i:i + a.chunk_rows] for i in range(0, nq, a.chunk_rows)]
        for out in run_pool(_rank_worker, chunks, a.lookup_workers):
            absorb(out)
        dt = time.time() - tl
        log(f"country={country}: lookup done in {dt:.0f}s ({nq / max(dt, 1e-9):,.0f} S1 rows/s)")
        log_mem(f"country={country}: after lookup")

        if a.tfidf_k > 0:
            top_a = res["a"][:, 0]
            top_s = res["s"][:, 0]
            weak = np.flatnonzero((res["n"] < a.tfidf_weak_threshold) |
                                  (top_a < a.tfidf_min_confident_agreement) |
                                  (top_s < a.tfidf_min_confident_score))
            log(f"country={country}: {len(weak):,} weak rows -> TF-IDF top-{a.tfidf_k} ...")
            if len(weak):
                tfl, tof = tfidf_hits_csr(list(Q["squashed"][weak]), list(corpus_sq), a.tfidf_k,
                                          a.tfidf_max_df, a.tfidf_threads)
                ln = np.zeros(nq, np.int64)
                ln[weak] = np.diff(tof)
                full_off = np.zeros(nq + 1, np.int64)
                np.cumsum(ln, out=full_off[1:])
                _SHARED["tfidf"] = (tfl, full_off)
                wchunks = [weak[i:i + a.chunk_rows] for i in range(0, len(weak), a.chunk_rows)]
                for out in run_pool(_rank_worker, wchunks, a.lookup_workers):
                    absorb(out)
                res["tfidf_ran"][weak] = True
                _SHARED["tfidf"] = None

        # ---- reverse (competing-S1) statistics over every S1 row of this country
        used_weights = _SHARED.get("weights") is not None
        _SHARED.clear()
        del idx, cname, caddr
        gc.collect()
        tr_ = time.time()
        rev = reverse_stats(res, len(corpus_ids))
        log(f"country={country}: reverse competing-S1 stats in {time.time() - tr_:.0f}s"
            + ("" if not a.sample_s1 else
               "  !! computed over a SAMPLE of S1 rows -- for training/test parquets use --sample-s1 0"))
        log_mem(f"country={country}: after reverse stats")

        # ---- write this country's rows (optionally only a random subset, AFTER exact reverse stats)
        valid = res["c"] >= 0
        if a.write_sample and a.write_sample < nq:
            keep_rows = np.zeros(nq, bool)
            keep_rows[np.random.RandomState(a.seed + 11).choice(nq, a.write_sample, replace=False)] = True
            flat_keep = np.repeat(keep_rows, valid.sum(1))
            rev = {k: v[flat_keep] for k, v in rev.items()}
            for k in ("c", "a", "s", "r", "m", "jn", "ja", "n", "h", "tfidf_ran"):
                res[k] = res[k][keep_rows]
            q_ids = q_ids.filter(pa.array(keep_rows))
            if gt is not None:
                trm = [x for x, k in zip(trm, keep_rows) if k]
                trb = [x for x, k in zip(trb, keep_rows) if k]
                heldout = heldout[keep_rows]
            nq = int(keep_rows.sum())
            valid = res["c"] >= 0
            log(f"country={country}: writing {nq:,} sampled rows (reverse stats used every S1 row)")
        lens = valid.sum(1).astype(np.int32)
        loff = np.zeros(nq + 1, np.int32)
        np.cumsum(lens, out=loff[1:])
        loff_a = pa.array(loff)
        cols = {
            "source1_entity_id": q_ids,
            "country": pa.array([country] * nq),
            "candidate_entity_ids": pa.ListArray.from_arrays(loff_a, corpus_ids.take(pa.array(res["c"][valid]))),
            "candidate_agreement": pa.ListArray.from_arrays(loff_a, pa.array(res["a"][valid].astype(np.int32))),
            "candidate_coarse_score": pa.ListArray.from_arrays(loff_a, pa.array(res["s"][valid])),
            "candidate_rank_score": pa.ListArray.from_arrays(loff_a, pa.array(res["r"][valid])),
            # per-candidate, parallel to candidate_entity_ids (list position = rank):
            # bit f set <=> key family f (order: see log line 'family bits') retrieved it
            "candidate_family_mask": pa.ListArray.from_arrays(loff_a, pa.array(res["m"][valid])),
            "candidate_jaccard_name": pa.ListArray.from_arrays(loff_a, pa.array(res["jn"][valid])),
            "candidate_jaccard_addr": pa.ListArray.from_arrays(loff_a, pa.array(res["ja"][valid])),
            # competing S1 rows (see reverse_stats): count and best scores among the OTHER S1 rows
            "candidate_rev_n": pa.ListArray.from_arrays(loff_a, pa.array(rev["rev_n"])),
            "candidate_rev_rank_score": pa.ListArray.from_arrays(loff_a, pa.array(rev["rev_r"])),
            "candidate_rev_jaccard_name": pa.ListArray.from_arrays(loff_a, pa.array(rev["rev_jn"])),
            "candidate_rev_jaccard_addr": pa.ListArray.from_arrays(loff_a, pa.array(rev["rev_ja"])),
            "n_before_prune": pa.array(res["n"]),
            "tfidf_ran": pa.array(res["tfidf_ran"]),
        }
        for j, fn in enumerate(fam_names):
            cols[f"hits_{fn}"] = pa.array(res["h"][:, j])
        if gt is not None:
            cols["true_ranks"] = pa.array([x.tolist() for x in trm], type=pa.list_(pa.int32()))
        else:
            cols["true_ranks"] = pa.array([[]] * nq, type=pa.list_(pa.int32()))
        tbl = pa.table(cols)
        if writer is None:
            writer = pq.ParquetWriter(a.out, tbl.schema)
        writer.write_table(tbl)

        if gt is not None:
            recall_report([trm[i] for i in np.flatnonzero(heldout)],
                          f"country={country} {'learned ranker' if used_weights else 'baseline'}"
                          + (" (held-out rows)" if not heldout.all() else ""), a.topk)
            if used_weights:
                recall_report([trb[i] for i in np.flatnonzero(heldout)],
                              f"country={country} baseline agreement ranking (same rows)", a.topk)
            all_tr_model.extend(trm)
            all_tr_base.extend(trb)
            all_heldout.append(heldout)
        log(f"country={country}: done in {time.time() - tc:.0f}s")
        del Q, res, rev, trm, trb, tbl, cols, corpus_ids
        _SHARED.clear()
        gc.collect()

    if writer is not None:
        writer.close()
    log(f"wrote {a.out}")
    if fitted_w:
        with open(a.save_ranker, "w") as f:
            json.dump(fitted_w, f, indent=1)
        log(f"saved ranker weights for {sorted(fitted_w)} -> {a.save_ranker}")
    if all_tr_model and len(all_heldout) > 1:
        ho = np.concatenate(all_heldout)
        recall_report([all_tr_model[i] for i in np.flatnonzero(ho)], "ALL COUNTRIES (chosen ranking)", a.topk)
    log(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()

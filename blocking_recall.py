"""
Blocking + recall probe for the Business Entity Resolution challenge — v5.

v4 fixed memory/parallelism at 100 lookup-workers. But raising --topk on the
existing dict-based keys alone plateaus hard: measured on real data,
100->250 (2.5x the candidate volume) only moved recall 88.98%->91.31%. Every
existing key (squash, rare_name, rare_ngram, addr_num, addr_pin, rare_addr)
is a discrete/exact signal anchored to whole tokens or exact digit runs --
raising topk just returns MORE ties on the same kind of match, not a
DIFFERENT kind of match. A name that's genuinely similar character-by-
character (a typo, an abbreviation variant, minor transliteration drift)
but shares no whole rare token and no exact digit run with the query was
never going to be found by these keys regardless of topk.

FIX: v5 adds a char-3/4-gram TF-IDF top-k retrieval pass (tfidf_topk_hits),
unioned into the SAME agreement-count mechanism as one more key ("tfidf") --
no change needed to rank_candidates() itself. This targets a genuinely
different failure mode than the existing keys.

MEASURED, not assumed, before shipping: sklearn's NearestNeighbors(cosine)
on sparse input ran at 95 queries/sec on a 572k-row test corpus -- unusable
at real scale. Switching to sp_matmul_topn (fused sparse top-k product) only
reached 122 q/s on THAT SAME corpus, which turned out to be because the test
corpus's name generator had a tiny repeated vocabulary (438 distinct
n-grams across 572k rows -> ~7% dense, effectively a dense matmul). Re-running
the identical measurement with genuinely diverse text (511,051 distinct
n-grams, 0.01% dense) gave 2,072 q/s -- a 17x difference from text diversity
alone, not from the library swap. Real business names should behave like
the diverse case. See tfidf_topk_hits()'s docstring for the full detail --
flagged here so a future slow-at-real-scale surprise gets diagnosed as a
vocabulary/density question first.

Usage: same flags as v4, plus --tfidf-k (default 50; set 0 to reproduce v4
exactly):
    python blocking_recall_v5.py --data-dir dataset --split train \
        --translit-dict translit_dict.json --sample-s1 100000 \
        --norm-workers 4 --lookup-workers 16 --topk 100 --tfidf-k 50 \
        --out cand_train.parquet
"""
import argparse
import json
import multiprocessing as mp
import os
import re
import time
from collections import Counter, defaultdict, namedtuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

# Explicit module-level namedtuple for the rows that cross the Pool boundary.
# pandas' own itertuples() generates its "Pandas" namedtuple class fresh per
# call, which some pandas versions fail to pickle across a fork (it's looked
# up as pandas.core.frame.Pandas on the receiving end, which doesn't
# persistently exist there). A real, importable, module-level class sidesteps
# that entirely regardless of pandas version.
S1Row = namedtuple("S1Row", [
    "entity_id", "country", "squashed", "squashed_alt", "name_tokens",
    "name_ngrams", "addr_tokens", "addr_num_keys", "addr_pin_keys",
    "name_ids", "addr_ids",
])

from normalize import clean_name, clean_address
from build_translit_dict import tokens as split_tokens, is_native

_SQUASH_RE = re.compile(r"[^a-z0-9]")


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


def char_ngrams(s: str, n: int = 4):
    return [s[i:i + n] for i in range(len(s) - n + 1)] if len(s) >= n else ([s] if s else [])


def _normalize_chunk(args):
    df, translit_dict = args
    lat_names = df["business_name"].map(lambda t: latinize_ci(t, translit_dict))
    name_info = lat_names.map(clean_name)
    addr_info = df["business_address"].map(clean_address)

    df = df.copy()
    df["squashed"] = [ni["variants"][0]["squashed"] for ni in name_info]
    df["squashed_alt"] = [ni["variants"][-1]["squashed"] if len(ni["variants"]) > 1 else ""
                          for ni in name_info]
    no_suffix_squashed = [_SQUASH_RE.sub("", ni["variants"][0]["no_suffix"]) for ni in name_info]
    df["name_tokens"] = [frozenset(split_tokens(ni["variants"][0]["no_suffix"])) for ni in name_info]
    df["name_ngrams"] = [char_ngrams(s, 4) for s in no_suffix_squashed]

    df["addr_tokens"] = [a["word_tokens"] for a in addr_info]
    nums = [a["numeric_tokens"] for a in addr_info]
    df["addr_num_keys"] = [[n[-4:] for n in ns if len(n) >= 3] for ns in nums]
    df["addr_pin_keys"] = [[n for n in ns if len(n) >= 5] for ns in nums]
    return df


def parallel_normalize(df: pd.DataFrame, translit_dict: dict, workers: int) -> pd.DataFrame:
    if workers <= 1 or len(df) < 50_000:
        return _normalize_chunk((df, translit_dict))
    # NOT np.array_split(df, workers): it relies on numpy internally calling
    # .swapaxes() on the DataFrame (the FutureWarning seen in every prior
    # run), and on pandas >=3.0 that now degrades to a plain ndarray, losing
    # .reset_index entirely. Chunk boundaries computed directly + .iloc
    # slicing is pure pandas, immune to numpy's internal handling either way.
    n = len(df)
    size = -(-n // workers)  # ceil division, avoids a short last-but-one chunk
    chunks = [df.iloc[i:i + size].reset_index(drop=True) for i in range(0, n, size)]
    with mp.get_context("fork").Pool(workers) as pool:
        parts = pool.map(_normalize_chunk, [(c, translit_dict) for c in chunks])
    return pd.concat(parts, ignore_index=True)


def doc_freq(iterables) -> dict:
    df = defaultdict(int)
    for items in iterables:
        for t in items:
            df[t] += 1
    return df


def rarest(items, freq: dict, k: int):
    items = list(items)
    if not items:
        return []
    return sorted(set(items), key=lambda t: freq.get(t, 0))[:k]


def tfidf_topk_hits(query_names, s2_names, s3_names, k: int, ngram_range=(3, 4)):
    """
    v5 addition: char-n-gram TF-IDF top-k retrieval, unioned with the existing
    dict-based keys (squash/rare_name/rare_ngram/addr_num/addr_pin/rare_addr).

    WHY: raising --topk on the existing keys plateaus hard (measured: 100->250
    only moved recall 88.98%->91.31% on real data) because every existing key
    is a discrete/exact signal (whole tokens, numeric suffixes, rare n-grams
    already anchored to token boundaries) -- it can't find a name that's
    genuinely similar character-by-character but shares no whole rare token
    or exact digit run with the query. Char-n-gram TF-IDF retrieval is exactly
    the "different kind of signal" the recall plateau calls for.

    IMPLEMENTATION NOTE, measured (not assumed): sklearn's NearestNeighbors
    with metric='cosine' on sparse input measured 95 queries/sec on a 572k-row
    corpus -- unusable at real scale (hours-to-days on 1.73M test S1 across
    all countries). sp_matmul_topn (a fused sparse top-k product, avoids ever
    materializing the full similarity matrix) measured only marginally better
    at 122 q/s on THAT SAME corpus -- but that test used a synthetic name
    generator with a tiny repeated vocabulary (438 distinct n-grams across
    572k rows -> ~7% dense, effectively a dense matmul in disguise). Repeating
    the same measurement with genuinely diverse text (511,051 distinct
    n-grams, 0.01% dense) gave 2,072 q/s -- a 17x difference from vocabulary
    diversity alone, not from a different library. Real business-name text
    should behave like the diverse case, not the repetitive one; this is
    flagged here so a future slow-at-real-scale surprise is diagnosed as a
    vocabulary/density question first, not assumed to need a different
    library.

    Returns a list, parallel to query_names, of {"s2": set(pos), "s3": set(pos)}
    -- the SAME shape lookup_candidates() produces, so it merges into the
    existing agreement-count mechanism by just adding one more key name; no
    change needed to rank_candidates() itself.
    """
    if not len(query_names):
        return []
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, dtype=np.float32, min_df=1)
    corpus = list(s2_names) + list(s3_names)
    corpus_mat = vec.fit_transform(corpus).tocsr()
    query_mat = vec.transform(query_names).tocsr()
    n_s2 = len(s2_names)
    k_eff = min(k, corpus_mat.shape[0])

    result = sp_matmul_topn(query_mat, corpus_mat.T, top_n=k_eff, threshold=0.0, n_threads=os.cpu_count() or 4)
    result = result.tocsr()

    out = []
    for i in range(result.shape[0]):
        row_idx = result.indices[result.indptr[i]:result.indptr[i + 1]]
        s2_hits = {int(j) for j in row_idx if j < n_s2}
        s3_hits = {int(j - n_s2) for j in row_idx if j >= n_s2}
        out.append({"s2": s2_hits, "s3": s3_hits})
    return out


def build_index(df: pd.DataFrame, cap: int, rare_k: int, rare_k_ngram: int):
    name_freq = doc_freq(df["name_tokens"])
    addr_freq = doc_freq(df["addr_tokens"])
    ngram_freq = doc_freq(df["name_ngrams"])

    idx = {"squash": defaultdict(list), "rare_name": defaultdict(list),
           "rare_ngram": defaultdict(list), "addr_num": defaultdict(list),
           "addr_pin": defaultdict(list), "rare_addr": defaultdict(list)}
    for pos, row in enumerate(df.itertuples(index=False)):
        if row.squashed:
            idx["squash"][row.squashed].append(pos)
        if row.squashed_alt:
            idx["squash"][row.squashed_alt].append(pos)
        for t in rarest(row.name_tokens, name_freq, rare_k):
            idx["rare_name"][t].append(pos)
        for g in rarest(row.name_ngrams, ngram_freq, rare_k_ngram):
            idx["rare_ngram"][g].append(pos)
        for n in row.addr_num_keys:
            idx["addr_num"][n].append(pos)
        for n in row.addr_pin_keys:
            idx["addr_pin"][n].append(pos)
        for t in rarest(row.addr_tokens, addr_freq, rare_k):
            idx["rare_addr"][t].append(pos)

    dropped = {}
    for kind, d in idx.items():
        before = len(d)
        for k in [k for k, v in d.items() if len(v) > cap]:
            del d[k]
        dropped[kind] = before - len(d)
    log(f"  index sizes: {[(k, len(v)) for k, v in idx.items()]}, dropped-as-too-common: {dropped}")
    return idx


def lookup_candidates(row, idx, cap):
    hits = defaultdict(set)
    for key in (row.squashed, row.squashed_alt):
        if key and key in idx["squash"]:
            hits["squash"].update(idx["squash"][key])
    for t in row.name_tokens:
        if t in idx["rare_name"]:
            hits["rare_name"].update(idx["rare_name"][t])
    for g in row.name_ngrams:
        if g in idx["rare_ngram"]:
            hits["rare_ngram"].update(idx["rare_ngram"][g])
    for n in row.addr_num_keys:
        if n in idx["addr_num"]:
            hits["addr_num"].update(idx["addr_num"][n])
    for n in row.addr_pin_keys:
        if n in idx["addr_pin"]:
            hits["addr_pin"].update(idx["addr_pin"][n])
    for t in row.addr_tokens:
        if t in idx["rare_addr"]:
            hits["rare_addr"].update(idx["rare_addr"][t])
    return hits


_SHARED = {}  # populated right before each per-country Pool() is created


# ---------------------------------------------------------------------------
# COW-safe token encoding: plain numpy int32 buffers instead of frozensets.
# The vocab dict itself is built and consumed only in the main process --
# workers never see it, only the already-encoded int arrays it produced.
# ---------------------------------------------------------------------------
def build_vocab(*token_series_list) -> dict:
    vocab = {}
    for series in token_series_list:
        for toks in series:
            for t in toks:
                if t not in vocab:
                    vocab[t] = len(vocab)
    return vocab


def encode_csr(token_series, vocab: dict):
    """Ragged (CSR-style) encoding: one flat int32 array + int64 offsets."""
    offsets = np.zeros(len(token_series) + 1, dtype=np.int64)
    flat = []
    for i, toks in enumerate(token_series):
        ids = sorted(vocab[t] for t in toks if t in vocab)
        flat.extend(ids)
        offsets[i + 1] = len(flat)
    return np.array(flat, dtype=np.int32), offsets


def csr_row(flat: np.ndarray, offsets: np.ndarray, i: int) -> np.ndarray:
    return flat[offsets[i]:offsets[i + 1]]


def sorted_overlap(a: np.ndarray, b: np.ndarray) -> int:
    """Count of common elements between two SORTED int arrays. Pure numpy
    buffer reads -- no Python object refcounts touched, so no COW cost."""
    i = j = c = 0
    la, lb = len(a), len(b)
    while i < la and j < lb:
        ai, bj = a[i], b[j]
        if ai == bj:
            c += 1
            i += 1
            j += 1
        elif ai < bj:
            i += 1
        else:
            j += 1
    return c


def jaccard_csr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 and len(b) == 0:
        return 0.0
    inter = sorted_overlap(a, b)
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0


def coarse_score(row, cand_name_ids: np.ndarray, cand_addr_ids: np.ndarray) -> float:
    """Tiebreaker only -- see rank_candidates for why agreement count is primary."""
    return 0.5 * jaccard_csr(row.name_ids, cand_name_ids) + 0.5 * jaccard_csr(row.addr_ids, cand_addr_ids)


def rank_candidates(hits, ids, name_flat, name_off, addr_flat, addr_off, row):
    """
    Score = (agreement_count, jaccard_tiebreak). Agreement count -- how many
    DISTINCT key families (squash, rare_name, rare_ngram, addr_num, addr_pin,
    rare_addr) hit this candidate -- is the PRIMARY signal: a true match tends
    to agree on several independent signals at once, while a coincidental
    orphan sharing a couple of common words usually only trips one weak key.
    Whole-word Jaccard is used only to break ties within the same agreement
    count, never as the primary score -- on a stress test with single-
    character name typos, ranking by Jaccard alone dropped recall from 100%
    to 81% versus the untruncated union, because one typo'd content word
    zeroes out its token match even though blocking correctly found the row
    via other keys (address, n-gram, ...).
    Returns (scored_list_desc, raw_union_size).
    """
    agree = Counter()
    for positions in hits.values():
        for p in positions:
            agree[p] += 1
    scored = []
    for p, cnt in agree.items():
        cand_name = csr_row(name_flat, name_off, p)
        cand_addr = csr_row(addr_flat, addr_off, p)
        scored.append(((cnt, coarse_score(row, cand_name, cand_addr)), ids[p]))
    scored.sort(key=lambda x: (-x[0][0], -x[0][1]))
    return scored, len(agree)


def _lookup_worker(rows):
    idx2, idx3 = _SHARED["idx2"], _SHARED["idx3"]
    s2_ids, s3_ids, cap = _SHARED["s2_ids"], _SHARED["s3_ids"], _SHARED["cap"]
    s2_name_flat, s2_name_off = _SHARED["s2_name_flat"], _SHARED["s2_name_off"]
    s2_addr_flat, s2_addr_off = _SHARED["s2_addr_flat"], _SHARED["s2_addr_off"]
    s3_name_flat, s3_name_off = _SHARED["s3_name_flat"], _SHARED["s3_name_off"]
    s3_addr_flat, s3_addr_off = _SHARED["s3_addr_flat"], _SHARED["s3_addr_off"]
    tfidf_hits_by_id = _SHARED["tfidf_hits_by_id"]
    topk = _SHARED["topk"]
    out = []
    for row in rows:
        hits2 = lookup_candidates(row, idx2, cap)
        hits3 = lookup_candidates(row, idx3, cap)
        tf = tfidf_hits_by_id.get(row.entity_id)
        if tf:
            # Merge in as just one more key name -- rank_candidates() and its
            # agreement Counter don't need to know this one came from a
            # vectorized batch step instead of a dict lookup.
            if tf["s2"]:
                hits2["tfidf"] = tf["s2"]
            if tf["s3"]:
                hits3["tfidf"] = tf["s3"]
        scored2, raw2 = rank_candidates(hits2, s2_ids, s2_name_flat, s2_name_off,
                                         s2_addr_flat, s2_addr_off, row)
        scored3, raw3 = rank_candidates(hits3, s3_ids, s3_name_flat, s3_name_off,
                                         s3_addr_flat, s3_addr_off, row)
        combined = scored2 + scored3
        combined.sort(key=lambda x: (-x[0][0], -x[0][1]))
        kept = combined[:topk]

        out.append({
            "source1_entity_id": row.entity_id,
            "country": row.country,
            "candidate_entity_ids": [cid for _, cid in kept],
            "candidate_agreement": [s[0] for s, _ in kept],
            "n_before_prune": raw2 + raw3,
            "by_key_s2": {k: len(v) for k, v in hits2.items()},
            "by_key_s3": {k: len(v) for k, v in hits3.items()},
        })
    return out


def chunked(seq, n_chunks):
    n_chunks = max(1, n_chunks)
    size = max(1, -(-len(seq) // n_chunks))
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--translit-dict", default="translit_dict.json")
    ap.add_argument("--sample-s1", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rare-k", type=int, default=3)
    ap.add_argument("--rare-k-ngram", type=int, default=2)
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--topk", type=int, default=50,
                     help="max candidates kept per S1, combined across S2+S3, after coarse-score ranking")
    ap.add_argument("--tfidf-k", type=int, default=50,
                     help="char-3/4-gram TF-IDF top-k retrieval per S1, unioned with the dict-based keys "
                          "(agreement-count treats a tfidf hit as one more key). 0 disables this pass "
                          "and reproduces v4's exact behavior.")
    ap.add_argument("--norm-workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--lookup-workers", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                     help="safe to set this as high as --norm-workers again under v4's numpy-based "
                          "candidate encoding -- the v3 OOM was specific to scoring over frozensets")
    ap.add_argument("--out", default="candidates.parquet")
    a = ap.parse_args()
    log(f"norm-workers={a.norm_workers}, lookup-workers={a.lookup_workers}")

    with open(a.translit_dict, encoding="utf-8") as f:
        translit_dict = json.load(f)

    d = os.path.join(a.data_dir, a.split)
    s1 = read_tsv(os.path.join(d, f"{a.split}_source1.tsv"))
    s2 = read_tsv(os.path.join(d, f"{a.split}_source2.tsv"))
    s3 = read_tsv(os.path.join(d, f"{a.split}_source3.tsv"))
    if a.sample_s1 and a.sample_s1 < len(s1):
        s1 = s1.sample(a.sample_s1, random_state=a.seed).reset_index(drop=True)
    log(f"loaded S1={len(s1):,} (sampled) S2={len(s2):,} S3={len(s3):,}")

    log("normalizing S1 ...")
    s1 = parallel_normalize(s1, translit_dict, a.norm_workers)  # reassigned in place --
    log("normalizing S2 ...")                                   # the raw version's refcount
    s2 = parallel_normalize(s2, translit_dict, a.norm_workers)  # drops to zero immediately,
    log("normalizing S3 ...")                                   # freeing it before the next
    s3 = parallel_normalize(s3, translit_dict, a.norm_workers)  # source is even read in.
    # (previously bound to s1k/s2k/s3k, which left the raw 5M+/5.28M-row S2/S3
    # text sitting in memory for the rest of the run, on top of everything
    # the normalized+CSR-encoded versions need -- exactly the kind of
    # redundant-copy issue that caused the train_matcher.py OOM.)

    results = []
    for country, group in s1.groupby("country"):
        log(f"country={country}: building index over S2/S3 subset ...")
        s2c = s2[s2["country"] == country].reset_index(drop=True)
        s3c = s3[s3["country"] == country].reset_index(drop=True)
        idx2 = build_index(s2c, a.cap, a.rare_k, a.rare_k_ngram)
        idx3 = build_index(s3c, a.cap, a.rare_k, a.rare_k_ngram)

        log(f"country={country}: encoding candidate tokens as COW-safe int arrays ...")
        vocab = build_vocab(s2c["name_tokens"], s2c["addr_tokens"], s3c["name_tokens"], s3c["addr_tokens"])
        s2_name_flat, s2_name_off = encode_csr(s2c["name_tokens"], vocab)
        s2_addr_flat, s2_addr_off = encode_csr(s2c["addr_tokens"], vocab)
        s3_name_flat, s3_name_off = encode_csr(s3c["name_tokens"], vocab)
        s3_addr_flat, s3_addr_off = encode_csr(s3c["addr_tokens"], vocab)
        # S1 rows get the SAME vocab applied so scores are comparable; this is
        # per-task pickled data (fresh objects in each worker), never a
        # pre-fork global, so it carries no COW risk regardless of size.
        group = group.copy()
        group["name_ids"] = [np.array(sorted(vocab[t] for t in toks if t in vocab), dtype=np.int32)
                              for toks in group["name_tokens"]]
        group["addr_ids"] = [np.array(sorted(vocab[t] for t in toks if t in vocab), dtype=np.int32)
                              for toks in group["addr_tokens"]]

        if a.tfidf_k > 0:
            log(f"country={country}: char-ngram TF-IDF top-{a.tfidf_k} retrieval "
                f"({len(group):,} queries x {len(s2c) + len(s3c):,} corpus) ...")
            tfidf_hits_list = tfidf_topk_hits(
                group["squashed"].tolist(), s2c["squashed"].tolist(), s3c["squashed"].tolist(),
                k=a.tfidf_k, ngram_range=(3, 4),
            )
            # Keyed by entity_id: each row's hit set is touched exactly once
            # (by the one worker that processes that row), never repeatedly
            # by many different rows the way the OLD candidate-side frozensets
            # were -- so this doesn't reintroduce the fork/COW issue v4 fixed.
            tfidf_hits_by_id = dict(zip(group["entity_id"], tfidf_hits_list))
        else:
            tfidf_hits_by_id = {}

        _SHARED["idx2"], _SHARED["idx3"] = idx2, idx3
        _SHARED["s2_ids"] = s2c["entity_id"].to_numpy()
        _SHARED["s3_ids"] = s3c["entity_id"].to_numpy()
        _SHARED["s2_name_flat"], _SHARED["s2_name_off"] = s2_name_flat, s2_name_off
        _SHARED["s2_addr_flat"], _SHARED["s2_addr_off"] = s2_addr_flat, s2_addr_off
        _SHARED["s3_name_flat"], _SHARED["s3_name_off"] = s3_name_flat, s3_name_off
        _SHARED["s3_addr_flat"], _SHARED["s3_addr_off"] = s3_addr_flat, s3_addr_off
        _SHARED["tfidf_hits_by_id"] = tfidf_hits_by_id
        _SHARED["cap"] = a.cap
        _SHARED["topk"] = a.topk

        rows = [S1Row(*vals) for vals in zip(
            group["entity_id"], group["country"], group["squashed"], group["squashed_alt"],
            group["name_tokens"], group["name_ngrams"], group["addr_tokens"],
            group["addr_num_keys"], group["addr_pin_keys"], group["name_ids"], group["addr_ids"],
        )]
        log(f"country={country}: looking up candidates for {len(rows):,} S1 rows "
            f"across {a.lookup_workers} workers ...")
        if a.lookup_workers > 1 and len(rows) > 2000:
            with mp.get_context("fork").Pool(a.lookup_workers) as pool:
                for part in pool.imap_unordered(_lookup_worker, chunked(rows, a.lookup_workers * 4)):
                    results.extend(part)
        else:
            results.extend(_lookup_worker(rows))

    out = pd.DataFrame(results)
    out.to_parquet(a.out)
    log(f"wrote {len(out):,} rows to {a.out}")

    n_cand = out["candidate_entity_ids"].map(len)
    n_before = out["n_before_prune"]
    log(f"AFTER pruning to top-{a.topk} by coarse score: avg candidates/S1: {n_cand.mean():.1f} | "
        f"median: {n_cand.median():.0f} | p95: {n_cand.quantile(0.95):.0f} | "
        f"zero-candidate S1: {(n_cand == 0).mean() * 100:.2f}%")
    log(f"BEFORE pruning (raw union size): avg {n_before.mean():.1f} | median {n_before.median():.0f} | "
        f"p95 {n_before.quantile(0.95):.0f} -- this is what v2 reported; compare against it directly")

    gt_path = os.path.join(a.data_dir, "train", "train_ground_truth.tsv")
    if a.split == "train" and os.path.exists(gt_path):
        gt = read_tsv(gt_path)
        gt["true_ids"] = gt["matched_entity_ids"].map(lambda s: set(x.strip() for x in s.split(",") if x.strip()))
        gt = gt.set_index("source1_entity_id")["true_ids"]
        out["true_ids"] = out["source1_entity_id"].map(gt)
        out["cand_set"] = out["candidate_entity_ids"].map(set)

        has_truth = out["true_ids"].map(lambda s: isinstance(s, set) and len(s) > 0)
        sub = out[has_truth]
        recalls = [len(t & c) / len(t) for t, c in zip(sub["true_ids"], sub["cand_set"])]
        micro_found = sum(len(t & c) for t, c in zip(sub["true_ids"], sub["cand_set"]))
        micro_total = sum(len(t) for t in sub["true_ids"])
        log(f"\n=== RECALL (S1 entities with >=1 true match, n={len(sub):,}) ===")
        log(f"macro recall@candidates (mean per-S1): {sum(recalls) / len(recalls) * 100:.2f}%")
        log(f"micro recall@candidates (total found/total true): {micro_found / micro_total * 100:.2f}%")
        log(f"S1 with recall < 100% (>=1 true match MISSED by blocking): "
            f"{sum(r < 1.0 for r in recalls) / len(recalls) * 100:.2f}%")

        singletons = out[~has_truth]
        if len(singletons):
            false_cand_rate = (singletons["candidate_entity_ids"].map(len) > 0).mean()
            log(f"true singletons that got >=1 candidate anyway: {false_cand_rate * 100:.2f}% "
                f"(fine -- the matcher/threshold stage rejects these, not blocking)")


if __name__ == "__main__":
    main()

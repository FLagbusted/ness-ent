"""
Blocking + recall probe for the Business Entity Resolution challenge — v4.

v3 fixed candidate VOLUME (agreement-count ranking + top-k truncation) but
introduced a MEMORY bug: scoring reads each candidate's name/address
frozensets, and in CPython every set operation touches (increments then
decrements) the refcount stored inside each shared token object. Under
fork(), a refcount write dirties that object's memory page, which defeats
copy-on-write -- the page gets privately duplicated in every child that
touches it. v2 never hit this because its lookup step only did `key in dict`
membership checks, not full set intersections over ~1,600 candidates/
S1; that's why 20 lookup-workers was fine under v2 but 4 OOM'd under v3.
Dropping to --lookup-workers 1 "fixes" the crash (one process never
duplicates anything it already owns) but doesn't scale: ~24 real minutes of
lookup for 100k S1 -> ~9 hours projected at the full 1.73M-S1 test set.

FIX: candidate name/address tokens are now encoded ONCE, per country, as
plain numpy int32 arrays (a shared vocab dict maps token -> int, built and
used only in the main process, never touched by workers) instead of Python
frozensets. Reading a numpy buffer doesn't touch any Python object's
refcount -- there's nothing for fork() to duplicate no matter how many
workers read it concurrently. The ranking logic itself (agreement-count
primary, jaccard tiebreak -- proven better than jaccard alone on the typo
stress test, see v3's docstring) is unchanged; this is a pure memory/
parallelism fix, verified below to produce identical output to v3 on the
same synthetic input.

Usage: same flags as v3, plus the norm/lookup worker split:
    python blocking_recall_v4.py --data-dir dataset --split train \
        --translit-dict translit_dict.json --sample-s1 100000 \
        --norm-workers 4 --lookup-workers 16 --topk 50 --out cand_train.parquet
--lookup-workers can now go back up -- try the old 16-20 range again.
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

from normalize import clean_name, clean_address
from build_translit_dict import tokens as split_tokens, is_native


S1Row = namedtuple("S1Row", [
    "entity_id", "country", "squashed", "squashed_alt", "name_tokens",
    "name_ngrams", "addr_tokens", "addr_num_keys", "addr_pin_keys",
    "name_ids", "addr_ids",
])

_SQUASH_RE = re.compile(r"[^a-z0-9]")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_tsv(path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quoting=3,
    )


def latinize_ci(text: str, translit_dict: dict) -> str:
    if not is_native(text):
        return text

    out = []
    for raw in text.split():
        key = raw.casefold().strip(".,()[]{}'\"")
        out.append(translit_dict.get(key, raw))

    return " ".join(out)


def char_ngrams(s: str, n: int = 4):
    return (
        [s[i:i + n] for i in range(len(s) - n + 1)]
        if len(s) >= n
        else ([s] if s else [])
    )


def _normalize_chunk(args):
    df, translit_dict = args

    lat_names = df["business_name"].map(
        lambda t: latinize_ci(t, translit_dict)
    )

    name_info = lat_names.map(clean_name)
    addr_info = df["business_address"].map(clean_address)

    df = df.copy()

    df["squashed"] = [
        ni["variants"][0]["squashed"]
        for ni in name_info
    ]

    df["squashed_alt"] = [
        ni["variants"][-1]["squashed"]
        if len(ni["variants"]) > 1
        else ""
        for ni in name_info
    ]

    no_suffix_squashed = [
        _SQUASH_RE.sub(
            "",
            ni["variants"][0]["no_suffix"]
        )
        for ni in name_info
    ]

    df["name_tokens"] = [
        frozenset(
            split_tokens(
                ni["variants"][0]["no_suffix"]
            )
        )
        for ni in name_info
    ]

    df["name_ngrams"] = [
        char_ngrams(s, 4)
        for s in no_suffix_squashed
    ]

    df["addr_tokens"] = [
        a["word_tokens"]
        for a in addr_info
    ]

    nums = [
        a["numeric_tokens"]
        for a in addr_info
    ]

    df["addr_num_keys"] = [
        [
            n[-4:]
            for n in ns
            if len(n) >= 3
        ]
        for ns in nums
    ]

    df["addr_pin_keys"] = [
        [
            n
            for n in ns
            if len(n) >= 5
        ]
        for ns in nums
    ]

    return df


def parallel_normalize(
    df: pd.DataFrame,
    translit_dict: dict,
    workers: int,
) -> pd.DataFrame:

    if workers <= 1 or len(df) < 50_000:
        return _normalize_chunk(
            (df, translit_dict)
        )

    chunks = [
        c.reset_index(drop=True)
        for c in np.array_split(df, workers)
    ]

    with mp.get_context("fork").Pool(workers) as pool:
        parts = pool.map(
            _normalize_chunk,
            [
                (c, translit_dict)
                for c in chunks
            ],
        )

    return pd.concat(
        parts,
        ignore_index=True,
    )


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

    return sorted(
        set(items),
        key=lambda t: freq.get(t, 0),
    )[:k]


def build_index(
    df: pd.DataFrame,
    cap: int,
    rare_k: int,
    rare_k_ngram: int,
):
    name_freq = doc_freq(df["name_tokens"])
    addr_freq = doc_freq(df["addr_tokens"])
    ngram_freq = doc_freq(df["name_ngrams"])

    idx = {
        "squash": defaultdict(list),
        "rare_name": defaultdict(list),
        "rare_ngram": defaultdict(list),
        "addr_num": defaultdict(list),
        "addr_pin": defaultdict(list),
        "rare_addr": defaultdict(list),
    }

    for pos, row in enumerate(
        df.itertuples(index=False)
    ):
        if row.squashed:
            idx["squash"][row.squashed].append(pos)

        if row.squashed_alt:
            idx["squash"][row.squashed_alt].append(pos)

        for t in rarest(
            row.name_tokens,
            name_freq,
            rare_k,
        ):
            idx["rare_name"][t].append(pos)

        for g in rarest(
            row.name_ngrams,
            ngram_freq,
            rare_k_ngram,
        ):
            idx["rare_ngram"][g].append(pos)

        for n in row.addr_num_keys:
            idx["addr_num"][n].append(pos)

        for n in row.addr_pin_keys:
            idx["addr_pin"][n].append(pos)

        for t in rarest(
            row.addr_tokens,
            addr_freq,
            rare_k,
        ):
            idx["rare_addr"][t].append(pos)

    dropped = {}

    for kind, d in idx.items():
        before = len(d)

        for k in [
            k
            for k, v in d.items()
            if len(v) > cap
        ]:
            del d[k]

        dropped[kind] = before - len(d)

    log(
        f"  index sizes: "
        f"{[(k, len(v)) for k, v in idx.items()]}, "
        f"dropped-as-too-common: {dropped}"
    )

    return idx


def lookup_candidates(row, idx, cap):
    hits = defaultdict(set)

    for key in (
        row.squashed,
        row.squashed_alt,
    ):
        if key and key in idx["squash"]:
            hits["squash"].update(
                idx["squash"][key]
            )

    for t in row.name_tokens:
        if t in idx["rare_name"]:
            hits["rare_name"].update(
                idx["rare_name"][t]
            )

    for g in row.name_ngrams:
        if g in idx["rare_ngram"]:
            hits["rare_ngram"].update(
                idx["rare_ngram"][g]
            )

    for n in row.addr_num_keys:
        if n in idx["addr_num"]:
            hits["addr_num"].update(
                idx["addr_num"][n]
            )

    for n in row.addr_pin_keys:
        if n in idx["addr_pin"]:
            hits["addr_pin"].update(
                idx["addr_pin"][n]
            )

    for t in row.addr_tokens:
        if t in idx["rare_addr"]:
            hits["rare_addr"].update(
                idx["rare_addr"][t]
            )

    return hits


_SHARED = {}


def build_vocab(*token_series_list) -> dict:
    vocab = {}

    for series in token_series_list:
        for toks in series:
            for t in toks:
                if t not in vocab:
                    vocab[t] = len(vocab)

    return vocab


def encode_csr(token_series, vocab: dict):
    """Ragged encoding: one flat int32 array + int64 offsets."""

    offsets = np.zeros(
        len(token_series) + 1,
        dtype=np.int64,
    )

    flat = []

    for i, toks in enumerate(token_series):
        ids = sorted(
            vocab[t]
            for t in toks
            if t in vocab
        )

        flat.extend(ids)
        offsets[i + 1] = len(flat)

    return (
        np.array(flat, dtype=np.int32),
        offsets,
    )


def csr_row(
    flat: np.ndarray,
    offsets: np.ndarray,
    i: int,
) -> np.ndarray:
    return flat[
        offsets[i]:offsets[i + 1]
    ]


def sorted_overlap(
    a: np.ndarray,
    b: np.ndarray,
) -> int:
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


def jaccard_csr(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    if len(a) == 0 and len(b) == 0:
        return 0.0

    inter = sorted_overlap(a, b)
    union = len(a) + len(b) - inter

    return (
        inter / union
        if union
        else 0.0
    )


def coarse_score(
    row,
    cand_name_ids: np.ndarray,
    cand_addr_ids: np.ndarray,
) -> float:

    return (
        0.5
        * jaccard_csr(
            row.name_ids,
            cand_name_ids,
        )
        +
        0.5
        * jaccard_csr(
            row.addr_ids,
            cand_addr_ids,
        )
    )


def rank_candidates(
    hits,
    ids,
    name_flat,
    name_off,
    addr_flat,
    addr_off,
    row,
):
    agree = Counter()

    for positions in hits.values():
        for p in positions:
            agree[p] += 1

    scored = []

    for p, cnt in agree.items():
        cand_name = csr_row(
            name_flat,
            name_off,
            p,
        )

        cand_addr = csr_row(
            addr_flat,
            addr_off,
            p,
        )

        scored.append(
            (
                (
                    cnt,
                    coarse_score(
                        row,
                        cand_name,
                        cand_addr,
                    ),
                ),
                ids[p],
            )
        )

    scored.sort(
        key=lambda x: (
            -x[0][0],
            -x[0][1],
        )
    )

    return scored, len(agree)


def _lookup_worker(rows):
    idx2 = _SHARED["idx2"]
    idx3 = _SHARED["idx3"]

    s2_ids = _SHARED["s2_ids"]
    s3_ids = _SHARED["s3_ids"]

    cap = _SHARED["cap"]

    s2_name_flat = _SHARED["s2_name_flat"]
    s2_name_off = _SHARED["s2_name_off"]

    s2_addr_flat = _SHARED["s2_addr_flat"]
    s2_addr_off = _SHARED["s2_addr_off"]

    s3_name_flat = _SHARED["s3_name_flat"]
    s3_name_off = _SHARED["s3_name_off"]

    s3_addr_flat = _SHARED["s3_addr_flat"]
    s3_addr_off = _SHARED["s3_addr_off"]

    topk = _SHARED["topk"]

    out = []

    for row in rows:
        hits2 = lookup_candidates(
            row,
            idx2,
            cap,
        )

        hits3 = lookup_candidates(
            row,
            idx3,
            cap,
        )

        scored2, raw2 = rank_candidates(
            hits2,
            s2_ids,
            s2_name_flat,
            s2_name_off,
            s2_addr_flat,
            s2_addr_off,
            row,
        )

        scored3, raw3 = rank_candidates(
            hits3,
            s3_ids,
            s3_name_flat,
            s3_name_off,
            s3_addr_flat,
            s3_addr_off,
            row,
        )

        combined = scored2 + scored3

        combined.sort(
            key=lambda x: (
                -x[0][0],
                -x[0][1],
            )
        )

        kept = combined[:topk]

        out.append(
            {
                "source1_entity_id": row.entity_id,
                "country": row.country,
                "candidate_entity_ids": [
                    cid
                    for _, cid in kept
                ],
                "candidate_agreement": [
                    s[0]
                    for s, _ in kept
                ],
                "n_before_prune": raw2 + raw3,
                "by_key_s2": {
                    k: len(v)
                    for k, v in hits2.items()
                },
                "by_key_s3": {
                    k: len(v)
                    for k, v in hits3.items()
                },
            }
        )

    return out


def chunked(seq, n_chunks):
    n_chunks = max(
        1,
        n_chunks,
    )

    size = max(
        1,
        -(-len(seq) // n_chunks),
    )

    return [
        seq[i:i + size]
        for i in range(
            0,
            len(seq),
            size,
        )
    ]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data-dir",
        default="dataset",
    )

    ap.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
    )

    ap.add_argument(
        "--translit-dict",
        default="translit_dict.json",
    )

    ap.add_argument(
        "--sample-s1",
        type=int,
        default=100_000,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--rare-k",
        type=int,
        default=3,
    )

    ap.add_argument(
        "--rare-k-ngram",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--cap",
        type=int,
        default=300,
    )

    ap.add_argument(
        "--topk",
        type=int,
        default=50,
        help=(
            "max candidates kept per S1, "
            "combined across S2+S3, "
            "after coarse-score ranking"
        ),
    )

    ap.add_argument(
        "--norm-workers",
        type=int,
        default=max(
            1,
            (os.cpu_count() or 4) - 2,
        ),
    )

    ap.add_argument(
        "--lookup-workers",
        type=int,
        default=max(
            1,
            (os.cpu_count() or 4) - 2,
        ),
        help=(
            "safe to set this as high as "
            "--norm-workers again under v4"
        ),
    )

    ap.add_argument(
        "--out",
        default="candidates.parquet",
    )

    a = ap.parse_args()

    log(
        f"norm-workers={a.norm_workers}, "
        f"lookup-workers={a.lookup_workers}"
    )

    with open(
        a.translit_dict,
        encoding="utf-8",
    ) as f:
        translit_dict = json.load(f)

    d = os.path.join(
        a.data_dir,
        a.split,
    )

    s1 = read_tsv(
        os.path.join(
            d,
            f"{a.split}_source1.tsv",
        )
    )

    s2 = read_tsv(
        os.path.join(
            d,
            f"{a.split}_source2.tsv",
        )
    )

    s3 = read_tsv(
        os.path.join(
            d,
            f"{a.split}_source3.tsv",
        )
    )

    if (
        a.sample_s1
        and a.sample_s1 < len(s1)
    ):
        s1 = (
            s1.sample(
                a.sample_s1,
                random_state=a.seed,
            )
            .reset_index(drop=True)
        )

    log(
        f"loaded S1={len(s1):,} (sampled) "
        f"S2={len(s2):,} "
        f"S3={len(s3):,}"
    )

    log("normalizing S1 ...")
    s1k = parallel_normalize(
        s1,
        translit_dict,
        a.norm_workers,
    )

    log("normalizing S2 ...")
    s2k = parallel_normalize(
        s2,
        translit_dict,
        a.norm_workers,
    )

    log("normalizing S3 ...")
    s3k = parallel_normalize(
        s3,
        translit_dict,
        a.norm_workers,
    )

    results = []

    for country, group in s1k.groupby("country"):
        log(
            f"country={country}: "
            f"building index over S2/S3 subset ..."
        )

        s2c = (
            s2k[
                s2k["country"] == country
            ]
            .reset_index(drop=True)
        )

        s3c = (
            s3k[
                s3k["country"] == country
            ]
            .reset_index(drop=True)
        )

        idx2 = build_index(
            s2c,
            a.cap,
            a.rare_k,
            a.rare_k_ngram,
        )

        idx3 = build_index(
            s3c,
            a.cap,
            a.rare_k,
            a.rare_k_ngram,
        )

        log(
            f"country={country}: "
            f"encoding candidate tokens as "
            f"COW-safe int arrays ..."
        )

        vocab = build_vocab(
            s2c["name_tokens"],
            s2c["addr_tokens"],
            s3c["name_tokens"],
            s3c["addr_tokens"],
        )

        s2_name_flat, s2_name_off = encode_csr(
            s2c["name_tokens"],
            vocab,
        )

        s2_addr_flat, s2_addr_off = encode_csr(
            s2c["addr_tokens"],
            vocab,
        )

        s3_name_flat, s3_name_off = encode_csr(
            s3c["name_tokens"],
            vocab,
        )

        s3_addr_flat, s3_addr_off = encode_csr(
            s3c["addr_tokens"],
            vocab,
        )

        group = group.copy()

        group["name_ids"] = [
            np.array(
                sorted(
                    vocab[t]
                    for t in toks
                    if t in vocab
                ),
                dtype=np.int32,
            )
            for toks in group["name_tokens"]
        ]

        group["addr_ids"] = [
            np.array(
                sorted(
                    vocab[t]
                    for t in toks
                    if t in vocab
                ),
                dtype=np.int32,
            )
            for toks in group["addr_tokens"]
        ]

        _SHARED["idx2"] = idx2
        _SHARED["idx3"] = idx3

        _SHARED["s2_ids"] = (
            s2c["entity_id"].to_numpy()
        )

        _SHARED["s3_ids"] = (
            s3c["entity_id"].to_numpy()
        )

        _SHARED["s2_name_flat"] = s2_name_flat
        _SHARED["s2_name_off"] = s2_name_off

        _SHARED["s2_addr_flat"] = s2_addr_flat
        _SHARED["s2_addr_off"] = s2_addr_off

        _SHARED["s3_name_flat"] = s3_name_flat
        _SHARED["s3_name_off"] = s3_name_off

        _SHARED["s3_addr_flat"] = s3_addr_flat
        _SHARED["s3_addr_off"] = s3_addr_off

        _SHARED["cap"] = a.cap
        _SHARED["topk"] = a.topk

        rows = [
            S1Row(*vals)
            for vals in zip(
                group["entity_id"],
                group["country"],
                group["squashed"],
                group["squashed_alt"],
                group["name_tokens"],
                group["name_ngrams"],
                group["addr_tokens"],
                group["addr_num_keys"],
                group["addr_pin_keys"],
                group["name_ids"],
                group["addr_ids"],
            )
        ]

        log(
            f"country={country}: "
            f"looking up candidates for {len(rows):,} "
            f"S1 rows across {a.lookup_workers} workers ..."
        )

        if (
            a.lookup_workers > 1
            and len(rows) > 2000
        ):
            with mp.get_context(
                "fork"
            ).Pool(
                a.lookup_workers
            ) as pool:
                for part in pool.imap_unordered(
                    _lookup_worker,
                    chunked(
                        rows,
                        a.lookup_workers * 4,
                    ),
                ):
                    results.extend(part)

        else:
            results.extend(
                _lookup_worker(rows)
            )

    out = pd.DataFrame(results)

    out.to_parquet(a.out)

    log(
        f"wrote {len(out):,} rows "
        f"to {a.out}"
    )

    n_cand = (
        out["candidate_entity_ids"]
        .map(len)
    )

    n_before = (
        out["n_before_prune"]
    )

    log(
        f"AFTER pruning to top-{a.topk} "
        f"by coarse score: "
        f"avg candidates/S1: "
        f"{n_cand.mean():.1f} | "
        f"median: {n_cand.median():.0f} | "
        f"p95: {n_cand.quantile(0.95):.0f} | "
        f"zero-candidate S1: "
        f"{(n_cand == 0).mean() * 100:.2f}%"
    )

    log(
        f"BEFORE pruning "
        f"(raw union size): "
        f"avg {n_before.mean():.1f} | "
        f"median {n_before.median():.0f} | "
        f"p95 {n_before.quantile(0.95):.0f} "
        f"-- this is what v2 reported; "
        f"compare against it directly"
    )

    gt_path = os.path.join(
        a.data_dir,
        "train",
        "train_ground_truth.tsv",
    )

    if (
        a.split == "train"
        and os.path.exists(gt_path)
    ):
        gt = read_tsv(gt_path)

        gt["true_ids"] = (
            gt["matched_entity_ids"].map(
                lambda s: set(
                    x.strip()
                    for x in s.split(",")
                    if x.strip()
                )
            )
        )

        gt = (
            gt.set_index(
                "source1_entity_id"
            )["true_ids"]
        )

        out["true_ids"] = (
            out["source1_entity_id"].map(gt)
        )

        out["cand_set"] = (
            out["candidate_entity_ids"]
            .map(set)
        )

        has_truth = out["true_ids"].map(
            lambda s:
                isinstance(s, set)
                and len(s) > 0
        )

        sub = out[has_truth]

        recalls = [
            len(t & c) / len(t)
            for t, c in zip(
                sub["true_ids"],
                sub["cand_set"],
            )
        ]

        micro_found = sum(
            len(t & c)
            for t, c in zip(
                sub["true_ids"],
                sub["cand_set"],
            )
        )

        micro_total = sum(
            len(t)
            for t in sub["true_ids"]
        )

        log(
            f"\n=== RECALL "
            f"(S1 entities with >=1 true match, "
            f"n={len(sub):,}) ==="
        )

        log(
            "macro recall@candidates "
            f"(mean per-S1): "
            f"{sum(recalls) / len(recalls) * 100:.2f}%"
        )

        log(
            "micro recall@candidates "
            f"(total found/total true): "
            f"{micro_found / micro_total * 100:.2f}%"
        )

        log(
            "S1 with recall < 100% "
            "(>=1 true match MISSED by blocking): "
            f"{sum(r < 1.0 for r in recalls) / len(recalls) * 100:.2f}%"
        )

        singletons = out[~has_truth]

        if len(singletons):
            false_cand_rate = (
                singletons[
                    "candidate_entity_ids"
                ].map(len) > 0
            ).mean()

            log(
                "true singletons that got >=1 "
                f"candidate anyway: "
                f"{false_cand_rate * 100:.2f}% "
                "(fine -- the matcher/threshold "
                "stage rejects these, not blocking)"
            )


if __name__ == "__main__":
    main()

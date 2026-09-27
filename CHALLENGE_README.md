# Business Entity Resolution — Amazon ML Challenge (Unstop)

Match noisy business records in S2/S3 to the clean reference entities in S1.
Scored by macro-averaged F0.5 per S1 entity (precision weighted 2x). Output:
`matching_results.tsv` + `candidate_pairs.tsv`. Model must be MIT/Apache-2.0,
≤8B params, no external data/APIs.

## Pipeline (run in this order)

```
normalize.py            -- text-cleaning primitives (no standalone run)
build_translit_dict.py  -- mines a Latin<->native-script token dictionary
                            from train ground truth (India, 9 scripts)
blocking_recall.py      -- candidate generation + recall probe (currently
                            v4 content: COW-safe int-array scoring, safe to
                            parallelize; agreement-count-primary ranking;
                            top-K pruning so candidate volume is tractable)
train_matcher.py         -- feature engineering + LightGBM classifier +
                            the challenge's own entity-level macro F0.5
                            evaluator with a threshold sweep
```

### 1. Mine the transliteration dictionary (once)

```bash
python build_translit_dict.py --data-dir dataset --out translit_dict.json
```

### 2. Blocking / candidate generation

```bash
python blocking_recall.py --data-dir dataset --split train \
  --translit-dict translit_dict.json --sample-s1 100000 \
  --norm-workers 4 --lookup-workers 16 --topk 100 \
  --out cand_train.parquet
```

`--topk` trades recall for candidate volume (avg candidates/S1 = topk,
roughly). `--lookup-workers` can be pushed as high as core count now that
candidate scoring uses raw numpy int arrays instead of Python frozensets
(the earlier fork/COW-refcount OOM is fixed at the source, not worked
around).

### 3. Feature engineering + LightGBM baseline

```bash
pip install rapidfuzz lightgbm pyarrow
python train_matcher.py --candidates cand_train.parquet --data-dir dataset \
  --translit-dict translit_dict.json --val-frac 0.2 \
  --out-model matcher.lgb --out-meta matcher_meta.json
```

Splits by S1 entity (not by pair) to avoid leakage, computes features fresh
from raw text (RapidFuzz ratios, name/address Jaccard, numeric-suffix
overlap for digit-drop corruption, domain/alias/phone flags, IDF-weighted
shared tokens, plus `agreement_count` reused from blocking), trains a
LightGBM binary classifier, then sweeps the decision threshold against the
challenge's *actual* per-entity F0.5 formula rather than AUC/accuracy.

## Real results so far (100k S1 sample, train split, India+US)

| Stage | Metric | Value |
|---|---|---|
| Blocking (topk=100) | macro recall@candidates | 88.98% |
| Blocking (topk=100) | micro recall@candidates | 88.90% |
| Blocking (topk=100) | avg candidates/S1 | 99.9 |
| Blocking (topk=100) | wall time, 16 lookup-workers | 29m32s |
| Matcher (candidates avg ~150/S1) | val pairwise AUC | 0.9993 |
| Matcher | best threshold | 0.95 |
| Matcher | val macro F0.5 (n=19,999 S1, incl. singletons) | 0.8927 |

Macro F0.5 landed close to the blocking recall ceiling — the classifier is
converting most of what blocking retrieves into correct decisions, so
**blocking recall, not model quality, is currently the binding constraint**
on further F0.5 gains.

## Full dataset (real, from `er_script_census.py`)

train S1=2,206,821 / S2=5,034,616 / S3=5,285,603
test  S1=1,732,544 / S2=4,887,273 / S3=5,082,316

Countries: train = US + India; test adds France (~15%, new accented-Latin
chars + legal/address vocab, zero train pairs to mine — hand-coded
normalization only, no dictionary). 9 Indic scripts. True pairs never cross
country labels — hard country partition is safe. Every S2/S3 record has at
most one S1 owner (0 exceptions, 7.6M matched ids) — not yet exploited.
5.6% of S1 are true singletons; ~3.5 avg matches/S1 (range 0–11); ~26% of
S2/S3 are genuine orphans/decoys.

## Hardware

- Desktop: HP Elite Tower 600 G9, Ubuntu, i7-13700 (24 threads), 32GB RAM,
  NVIDIA T400 4GB — CPU-heavy blocking/features/LightGBM.
- Jetson AGX Orin 64GB dev kit — reserved for an optional embedding /
  small-LLM-judge stage later (large unified memory for big batches).
- AWS, $100 credit — reserve burst capacity only (g4dn.xlarge/T4 ≈190hrs,
  g5.xlarge/A10G ≈99hrs on-demand). Budget alert set; always stop instances.

## Not yet built (roughly in priority order)

1. **Reverse-assignment / one-owner rule.** Ground truth confirms every
   S2/S3 record belongs to at most one S1 entity. Not used anywhere yet —
   likely the highest-leverage remaining feature/post-processing step given
   the model is already near the blocking-recall ceiling.
2. Raise blocking recall further (topk sweep, or a smarter candidate-rank
   score) — currently the binding constraint on F0.5.
3. Singleton threshold tuning / calibration.
4. France-specific robustness (leave-one-country-out validation; France has
   zero train pairs, so this is untested).
5. Full-scale run (2.2M S1 train, 1.73M S1 test) — current numbers are all
   on a 100k S1 sample.
6. Optional stage-2: cross-encoder or small MIT/Apache ≤8B LLM judge on the
   uncertain probability band (candidate: run on the Jetson for its 64GB
   unified memory).
7. Docker/submission packaging: `code/business_entity_resolution/src/`,
   `output/matching_results.tsv` + `output/candidate_pairs.tsv`, validated
   against `utils/validate_submission.py`.

## Known gotchas

- pandas' `itertuples()` row class isn't reliably picklable across a
  `multiprocessing.Pool` fork on every pandas version — `blocking_recall.py`
  uses an explicit module-level `namedtuple` instead.
- `rarest()`'s tie-breaking depends on Python's per-process string hash
  seed (randomized by default) — set `PYTHONHASHSEED` for reproducible
  before/after comparisons when testing changes.
- Jetson (aarch64): `rapidfuzz`/`lightgbm` install from prebuilt wheels via
  plain `pip`. If LightGBM fails to import with `libgomp.so.1: cannot open
  shared object file`, `sudo apt-get install -y libgomp1`.

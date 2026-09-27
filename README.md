# ness-ent: business entity resolution (S1 → S2 / S3)

For every Source-1 business record, find every Source-2 and Source-3 record of the same business (0..n matches).
The metric is macro F0.5 per S1 row: a row with no true match scores 1 only if we predict nothing, and precision
counts four times as much as recall.

The pipeline is classical and CPU-only. It uses no external data, no API lookups and no pretrained language model.
The only learned models are LightGBM gradient-boosted trees (MIT licence) and one logistic ranker, all trained on
the provided training labels.

## Pipeline

```
raw TSVs ──► normalize.py ──► blocking_recall.py ──► er_features.py ──► stage-1 LightGBM ──► stage-2 LightGBM ──► decision ──► owner-exclusive ──► matching_results.tsv
              (clean names,    (candidate keys,       (97 pair           (per-pair p)        (row/anchor           (expected      (each S2/S3 record
               addresses,       learned ranker,        features)                              consensus)            F0.5 per row)   to one S1 only)
               transliteration) top-200 per S1)
```

| Layer | File | What it does |
|---|---|---|
| Normalization | `normalize.py`, `build_translit_dict.py` | Folds accents and strips legal suffixes (Pvt Ltd, LLC, SARL…). Separates alias names ("f/k/a", "d.b.a."), web domains and phone suffixes. Latinizes Indic scripts using a token dictionary mined only from training positives. |
| Blocking | `blocking_recall.py` | Nine key families (squashed name, rare name tokens, rare character n-grams, address numbers, PIN / ZIP codes, rare address words, name+address pairs, …) are indexed as CSR arrays. A logistic ranker fitted on train orders the hits and keeps the top 200 per S1 row; the last 20 slots are reserved for exact-name hits. Also computes **reverse features**: how many *other* S1 rows claim the same record, and how strongly. |
| Features | `er_features.py` | 97 vectorized pair features in these groups: blocking scores, name strings (rapidfuzz), address strings, numbers (suffix / prefix / zero-strip equality), IDF-weighted token-set similarity with soft Jaro-Winkler token matching, name frequency, the reverse / competition features, and row-level context. |
| Stage 1 | `train_matcher.py` | LightGBM on all positives plus hard negatives (top-ranked non-matches) and a weighted random share of the other negatives. Three-seed ensemble. |
| Stage 2 | `train_matcher.py` | A second LightGBM that sees out-of-fold stage-1 probabilities plus the row's consensus: agreement with the row's top anchors and the probability landscape of the row. |
| Decision | `tune_decision.py`, `redecide.py` | Per row, choose the number of matches that maximizes expected F0.5 (with isotonic calibration); tuned by 2-fold CV on validation rows. Per-country overrides (`--country-rule France:min_p=0.7`) apply to countries not seen in training. |
| Owner-exclusive | `predict_matches.py` | Every S2/S3 record belongs to at most one S1 in the ground truth, so a record claimed by several S1 rows is kept only for the highest-probability claim. |

## Reproduce

```bash
pip install -r requirements.txt
# dataset/ holds train/ and test/ from the challenge (not in this repo)

# 1. transliteration dictionary (from training positives only)
python build_translit_dict.py --data-dir dataset --out translit_dict.json

# 2. ranker weights + training candidates (one run per country; PYTHONHASHSEED is fixed internally)
python blocking_recall.py --data-dir dataset --split train --translit-dict translit_dict.json --countries US \
  --topk 200 --reserve-exact 20 --fit-ranker 50000 --save-ranker ranker_us.json --write-sample 200000 \
  --out cand_train_us_R20.parquet
python blocking_recall.py --data-dir dataset --split train --translit-dict translit_dict.json --countries India \
  --topk 200 --reserve-exact 20 --fit-ranker 50000 --save-ranker ranker_india.json --write-sample 200000 \
  --out cand_train_india_R20.parquet

# 3. test candidates (France has no ranker of its own -> mean of US and India weights)
for c in US India France; do
  python blocking_recall.py --data-dir dataset --split test --translit-dict translit_dict.json --countries $c \
    --topk 200 --reserve-exact 20 --ranker-weights ranker_us.json,ranker_india.json \
    --out cand_test_${c,,}_R20.parquet
done

# 4. train (stage 1 ensemble + stage 2 + decision rule)
python train_matcher.py --candidates cand_train_us_R20.parquet,cand_train_india_R20.parquet \
  --data-dir dataset --translit-dict translit_dict.json --max-s1 400000 --val-rows 20000 \
  --lr 0.05 --n-models 3 --workers 12 --out-dir matcher

# 5. tune the decision on the rebuilt validation rows
python tune_decision.py --candidates cand_train_us_R20.parquet,cand_train_india_R20.parquet \
  --data-dir dataset --translit-dict translit_dict.json --model-dir matcher --max-s1 400000 \
  --val-rows 20000 --workers 12 --out decision.json

# 6. predict (streams 30k S1 rows per batch) and apply the decision
python predict_matches.py --candidates cand_test_us_R20.parquet,cand_test_india_R20.parquet,cand_test_france_R20.parquet \
  --data-dir dataset --split test --translit-dict translit_dict.json --model-dir matcher \
  --out-dir output --workers 12 --save-scores 1
python redecide.py --scores-dir output --out-dir submission --decision decision.json \
  --country-rule France:min_p=0.7
cp output/candidate_pairs.tsv submission/
```

Adjust `--workers` to your CPU. On a 30 GB machine, keep `--batch-s1` at 30000 or lower.

## Diagnostics

| Script | Purpose |
|---|---|
| `density_check.py` | Train vs test competition density of the reverse features (share of records with a competing S1, share "beaten"). |
| `tune_decision.py --exclude-seen … --compare-decision …` | Scores a model on rows from another candidates file, e.g. a density-matched resample, with a per-country breakdown of true vs predicted empty rate and matches per row. |
| `order_check.py` | Checks whether file order or ID numbers carry entity information. |
| `predict_matches.py` output | `diagnostics.txt` (contention by country, feature drift for unseen countries) and `contested_examples.txt`. |
| `train_matcher.py` report | Loss ledger: 1 − F0.5 split exactly into blocking misses, matcher recall, false positives, singleton merges. |

## Results (validation = 20k held-out training S1 rows; LB = public leaderboard)

| Version | Change | Val F0.5 | LB |
|---|---|---|---|
| v2 | blocking + single LightGBM | 0.9541 | 0.948 |
| v3 | stage-2 consensus model | 0.9601 | |
| v5 | reverse / competition features | 0.9733 | |
| v7 | IDF + soft token features | 0.9767 | |
| v9 | reserve-exact 20, 400k rows, 3-seed ensemble | 0.9799 | 0.9704 |
| v9 + France `min_p=0.7` | stricter decision for the unseen country | | 0.9709 |

Constraints respected: no external data or lookups; every model is LightGBM or logistic regression (MIT / BSD),
far below the 8B-parameter limit; the pipeline is deterministic for fixed seeds.

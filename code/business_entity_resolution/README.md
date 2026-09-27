# Business Entity Resolution — pipeline

Run every command from this folder (`code/business_entity_resolution/`). A CUDA GPU is used
for encoder training and nearest-neighbour search (tested on an 8 GB RTX 5060 laptop GPU with
PyTorch 2.11 + CUDA 12.8); 32 GB RAM and ~40 GB free disk are recommended. End to end the
pipeline takes about 7–8 hours on that machine.

No external data, APIs or pretrained models are used: every model is trained from scratch on
the provided data.

## Data location

By default the pipeline expects the repo layout:

- raw TSVs in `<repo>/student_resource/dataset/{train,test}/`
- caches in `<repo>/data_cache/`
- outputs in `<repo>/output/`

Override with `ER_DATA_DIR`, `ER_CACHE_DIR`, `ER_OUTPUT_DIR` (see `src/er/config.py`).

## Reproduce end to end

```bash
pip install -r requirements.txt     # torch: pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
bash run_all.sh
```

`run_all.sh` runs these stages in order (approximate times on the machine above):

| # | Stage | Commands | Time |
|---|---|---|---|
| 1 | Data preparation | `prepare_data.py`, `normalize_data.py`, `embed_data.py`, `featurize_data.py` | ~10 min |
| 2 | Base run | `train_encoder.py`, `retrieve.py` (train, test), `build_features.py`, `train_matcher.py`, `train_stage2.py`, `predict.py` | ~1.5 h |
| 3 | Self-training for France | `make_pseudo.py`, `train_encoder.py --pseudo … --tag _st`, `retrieve.py --tag _st` (train, test), `build_features.py --tag _st` | ~1 h |
| 4 | Cross-fitted training rows | per fold 0, 1: `train_encoder.py --exclude-fold f`, `retrieve.py --tag _ff`, `build_features.py --fold f` | ~1.2 h |
| 5 | v13 matcher (22M cross-fitted rows) | `v13/build_sidecars.py`, `v13/build_sidecars_v13.py`, `v13/train_v13.py`, `v13/predict_v13.py` (tuned settings in `configs/hp_results.json`) | ~2.5 h |
| 6 | France mix + blend | `v13/france_mix.py --cat-weight 0.6`, `ensemble_predict.py --weights 0.1 0.9 --adapted-for-unlabelled` | ~20 min |

The earlier v11 path (`train_big.py --out-tag _v11`, `blend_weight.py`, `france_veto.py`) is kept in
`src/` because the v11 → v13 comparisons in `../../FINDINGS.md` refer to it; the final run does not need it.

Validate with the official validator from `student_resource/`:

```bash
python utils/validate_submission.py --matching ../output/matching_results.tsv --candidate ../output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

**Reproducibility note.** The submitted files were produced stage by stage during the
competition with the same scripts; the base run used in the final ensemble was trained with an
earlier revision of the feature code (fewer features). Running `run_all.sh` with this code
therefore gives an equivalent submission (same pipeline, same hold-out behaviour), not a
byte-identical file. All intermediate scores are cached in `data_cache/model/`.

## Pipeline

| Stage | What it does |
|---|---|
| Normalise | `anyascii` transliteration (Indic scripts), mojibake repair, dotted-initial merging ("S.A.S." → sas), legal-suffix removal, state/region pulled into its own field (US/India codes, native-script names, French departments), address abbreviation canonicalisation. Country-agnostic. |
| Retrieve | Per country, each S2/S3 record retrieves S1 candidates from four retrievers: a learned two-tower encoder (hashed char 3-grams, trained contrastively from scratch) top-16, plus SVD char-TF-IDF combined / name / address top-4. Exact GPU top-k with block-max pruning. |
| Prune | Keep encoder top-5 + top-1 of each SVD retriever (~6.6 per record); records without an address keep all retrieved candidates. Hold-out pair recall 0.988. |
| Stage 1 | LightGBM over 66 pairwise features: retrieval cosines/ranks, competition context per record and per S1, name-key frequencies, name/address string similarities, house-number overlap/conflict/additions/typos, state agreement, legal-form and filler-word differences, and whether the record introduces a genuine name word or a look-alike marker. |
| Stage 2 | LightGBM on stage-1 features + stage-1 probability + per-S1 cluster agreement (similarity of the record to the other records stage 1 links to the same S1). |
| Training data | Cross-fitted: train S1 are split into folds; each fold's candidate pairs are scored by an encoder that never trained on that fold, so encoder features behave as on test. 22M training pairs. |
| France | The encoder is also trained on confident pseudo-labelled French pairs (self-training). After the decision, French pairs whose record introduces a genuine French name word or a look-alike marker are vetoed (`france_veto.py`). |
| Ensemble | Base run and big-data run averaged 0.2 / 0.8 over the union of candidates; countries without training labels use the adapted run only. |
| Decide | Each S2/S3 record goes to at most one S1 (ground truth never links a record to two): its highest-probability candidate if p >= 0.5. Each S1 then keeps the prefix of its matches that maximises expected F0.5 (empty list scored by P(singleton)). |

All learned stages are evaluated on a fixed 5% S1 hold-out (`er/holdout.py`) that no encoder or
matcher trains on.

## Results

| Version | Hold-out macro F0.5 | Public leaderboard |
|---|---|---|
| Exact name + house number rule | 0.655 (full train) | — |
| Stage-1 LightGBM + expected-F0.5 selection | 0.977 | — |
| + stage-2 cluster matcher | 0.982 | 0.971 |
| + France self-training | 0.982 | 0.972 |
| + cross-fitted training data, weighted ensemble | 0.985 | 0.9756 |
| + French name-word veto | 0.985 | 0.9773 |
| + v11 features (introduced words / numbers) | 0.9853 | 0.9784 |
| + v12 features (legal canon, signed number delta, added-word number share), no veto | 0.9870 | 0.979596 |
| + v13 features, tuned LightGBM, CatBoost stage 2 (0.3) | 0.98747 | 0.981542 |
| + France CatBoost weight 0.6 (France rows only) | 0.98747 | **0.981687** |

## Use of unlabelled test inputs

No test labels exist or are used. The pipeline does use the unlabelled test records in three
label-free ways: the SVD vectoriser is fitted on a sample of train + test text; the encoder is
self-trained on the model's own confident test predictions for France; and per-country word
statistics (name vocabulary, generator words) are counted on the split being scored.

## v13 (final): generator-recipe features + tuned LightGBM + CatBoost stage 2

Scripts in `src/v13/` (run from this folder after stages 1–5 above; intermediates in `data_cache/v13/`):

```bash
python src/v13/build_sidecars.py        # v12 columns + per-country added-word stats (train, test)
python src/v13/build_sidecars_v13.py    # v13 columns + word stats
python src/v13/hp_tune.py               # LightGBM / CatBoost settings on an S1-grouped hold-out slice
python src/v13/train_v13.py             # stage 1 LightGBM, stage 2 LightGBM + CatBoost (weight on hold-out)
python src/v13/predict_v13.py           # test scores -> data_cache/model/test_scores_v13.parquet
python src/ensemble_predict.py --scores test_scores.parquet test_scores_v13.parquet --weights 0.1 0.9 --tau 0.5 --adapted-for-unlabelled
```

New label-free, country-agnostic features: canonical legal forms (corp = corporation, SARL vs SA a real change),
signed house-number delta (look-alikes use the S1 number +1..+11), per-country added-word number-share rate
(look-alike markers keep the S1 house number far less often than generator noise words), dropped-word share,
street equality and first-number delta. Hold-out macro F0.5: v11 0.9853 → v12 0.9870 → v13 0.98747
(stage 2 = 0.7 LightGBM + 0.3 CatBoost). Model comparison on one hold-out slice: LightGBM 0.9811, CatBoost 0.9811,
XGBoost 0.9807, sklearn HistGB 0.9808, RandomForest 0.9742; LightGBM+CatBoost average 0.9815.

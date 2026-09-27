# Business Entity Resolution — Amazon ML Challenge 2026

Matches Source-2 / Source-3 business records to Source-1 entities (US, India, and France without
labels). Scored by macro F0.5 per S1 entity. **Best public leaderboard: 0.981687** (hold-out 0.98747).

- **[FINDINGS.md](FINDINGS.md)**: what moved the score (with leaderboard deltas), what didn't, data traps,
  how the synthetic look-alikes are made, and what we learned about France. Read this first.
- **[code/business_entity_resolution/](code/business_entity_resolution/)**: the full pipeline, with
  `run_all.sh` reproducing the best file end to end.
- **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**: the write-up submitted with the solution.

## Pipeline in one paragraph

Normalise (transliteration, mojibake repair, legal forms, state / house-number fields) → learned
two-tower char-3-gram encoder + SVD TF-IDF retrieval → stage-1 LightGBM on ~80 pairwise features
(string similarity, competition context, house-number and legal-form changes, generator-recipe
features) → stage-2 LightGBM + CatBoost with per-S1 cluster features → each record to at most one
S1, then per-S1 expected-F0.5 selection. Training rows are cross-fitted (22M pairs). France: the
encoder is self-trained on confident French pseudo-pairs, and French rows use more CatBoost weight.

## Run

```bash
cd code/business_entity_resolution
pip install -r requirements.txt   # torch: pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
bash run_all.sh                   # ~9 h on an 8 GB laptop GPU, 32 GB RAM, ~60 GB disk
```

The competition data is **not** in this repository. Put the organisers' files in
`student_resource/dataset/{train,test}/`, or point `ER_DATA_DIR` at them (see `src/er/config.py`).
Caches go to `data_cache/` and the submission to `output/`.

## Layout (under `code/business_entity_resolution/`)

| Path | Contents |
|---|---|
| `src/er/` | shared library: normalisation, features, retrieval, cluster features, decision, metric |
| `src/*.py` | pipeline stages 1–4 (data, encoder, retrieval, features, base matcher, cross-fitting) |
| `src/v13/` | final matcher: v12 / v13 features, training, prediction, France model mix |
| `src/france_veto.py` | French name-word veto (+0.0018 on models without recipe features) |
| `configs/hp_results.json` | tuned LightGBM / CatBoost settings used by `train_v13.py` |

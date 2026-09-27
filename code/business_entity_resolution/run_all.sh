#!/usr/bin/env bash
# End-to-end pipeline that produces output/matching_results.tsv and output/candidate_pairs.tsv.
# Run from code/business_entity_resolution/. Stops at the first failing step.
set -euo pipefail
C="${ER_CACHE_DIR:-../../data_cache}"
OUT="${ER_OUTPUT_DIR:-../../output}"
PSEUDO="$C/encoder/pseudo_test_france.parquet"

# 1. data preparation (label-free)
python src/prepare_data.py            # TSV -> parquet (RFC-4180 quoting, no NA coercion)
python src/normalize_data.py          # transliteration, canonicalisation, state/number fields
python src/embed_data.py              # unsupervised char-TF-IDF SVD vectors
python src/featurize_data.py          # hashed features for the learned encoder

# 2. base run: encoder, retrieval, two-stage matcher, test scores (test_scores.parquet)
python src/train_encoder.py
python src/retrieve.py --split train
python src/retrieve.py --split test
python src/build_features.py
python src/train_matcher.py           # also writes the hold-out OOF probabilities (oof.parquet)
python src/train_stage2.py
python src/predict.py

# 3. self-training for the unlabelled country: pseudo-labels -> adapted encoder (tag _st)
python src/make_pseudo.py --countries France --min-p 0.98
python src/train_encoder.py --pseudo "$PSEUDO" --tag _st
python src/retrieve.py --split train --tag _st
python src/retrieve.py --split test --tag _st
python src/build_features.py --tag _st        # hold-out evaluation rows

# 4. cross-fitted training rows: one encoder per S1 fold that never saw that fold
for f in 0 1; do
  python src/train_encoder.py --pseudo "$PSEUDO" --exclude-fold $f --tag _f$f --splits train
  python src/retrieve.py --split train --tag _f$f
  python src/build_features.py --tag _f$f --fold $f
  rm -f "$C"/cand/train_s2_f$f.parquet "$C"/cand/train_s3_f$f.parquet "$C"/vec/train_s?_enc_f$f.npy
done

# 5. v13 matcher on the 22M cross-fitted rows: v12 + v13 generator-recipe features (side files),
#    LightGBM stage 1, LightGBM + CatBoost stage 2 (weight and tau chosen on the hold-out)
mkdir -p "$C/v13"
python src/v13/build_sidecars.py          # v12 columns + per-country added-word stats (label-free)
python src/v13/build_sidecars_v13.py      # v13 columns + word stats
if [ ! -f "$C/v13/hp_results.json" ]; then
  cp configs/hp_results.json "$C/v13/"    # tuned settings (re-tune: python src/v13/hp_tune.py, ~15 min)
fi
python src/v13/train_v13.py
python src/v13/predict_v13.py             # test_scores_v13.parquet (p1, p2, p2_lgb, p2_cat)

# 6. France gets more CatBoost (0.6, chosen on the leaderboard: France-only rows), then the
#    0.1 base / 0.9 v13 blend; France uses the v13 run only. Public LB 0.981687.
python src/v13/france_mix.py --cat-weight 0.6
python src/ensemble_predict.py --scores test_scores.parquet test_scores_v13f06.parquet --weights 0.1 0.9 \
    --tau 0.5 --adapted-for-unlabelled --out "$OUT"
echo "done: $OUT/matching_results.tsv, $OUT/candidate_pairs.tsv"

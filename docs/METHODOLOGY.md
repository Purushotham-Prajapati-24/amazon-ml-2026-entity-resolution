# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We resolve S2/S3 records to Source-1 entities with a retrieve-then-classify pipeline: a
two-tower character-n-gram encoder trained from scratch on the provided pairs (plus
unsupervised char-TF-IDF retrievers) proposes ~6 S1 candidates per record on the GPU; a
two-stage LightGBM matcher scores them, the second stage checking each record against the
other records already linked to the same S1; a decision layer assigns every record to at
most one S1 and picks, per S1, the match set that maximises expected F0.5. The matchers are
trained on 22M cross-fitted pairs (features from encoders that never saw those entities) and
use features describing what a record *introduces* relative to the S1 (new house numbers,
genuine name words, look-alike markers). The encoder is adapted to the unlabelled French slice
by self-training, two runs built on different encoders are ensembled, and a French name-word
veto removes look-alikes. The final version (v13) adds label-free "generator recipe" features
(canonical legal forms, signed house-number delta, per-country added/dropped-word number-share
rates), tuned LightGBM settings and a CatBoost second-stage model averaged with LightGBM. On a
held-out 5% of training S1 entities the pipeline reaches **macro F0.5 0.9875** (precision
0.9978, recall 0.9672); the public leaderboard score is **0.9815**.

---

## 2. Methodology

### 2.1 Problem Analysis

Key findings from EDA on the provided data (no external data was used anywhere):

- **Scale:** train S1 2.21M, S2 5.03M, S3 5.29M; test S1 1.73M, S2 4.89M, S3 5.08M.
- **Assignment structure:** in the training ground truth every S2/S3 record matches **at most
  one** S1 (0 exceptions in 7.64M pairs) and no pair crosses countries. 5.6% of S1 entities
  are singletons; the mean is 3.46 matches (0–11). ~26% of S2/S3 records match nothing.
- **Weak identifiers:** postal codes are almost absent (US ~10%, India 0%, France <1%);
  38% of S1 names are repeated (chains), so names alone are ambiguous.
- **Scripts and formats:** 23.5% of India S2 names (13% in S3) and ~23% of India addresses
  are in Indic scripts; S3 spells US states in full but abbreviates Indian states (MH, WB);
  ~3% of S2/S3 names are URLs and ~3% of addresses are empty or `NULL`/`N/A`.
- **Noise:** typos and digit-for-letter swaps (`R0cky`), added diacritics, token reordering,
  wrong abbreviation expansion (`Street`→`Saint`), dotted legal forms (`S.A.S.`), legal-suffix
  changes, appended noise words (`Services`, `Center`; in France `& Fils`, `Associés`).
- **Hard negatives:** the unmatched records include generated look-alikes of real entities:
  same core name and a near-identical address but a different house number, a changed legal
  form, a one-letter name mutation, a swapped-in genuine name word, or an added marker
  ("Group"/"Holdings"; in France "Groupe"/"Holding"/"Participations").
- **Distribution shift:** France appears only in test (15% of test S1); ~800 test names use
  RFC-4180 quoting; some text carries UTF-8 mojibake.

### 2.2 Solution Strategy

**Approach Type:** Retrieval (learned + lexical blocking) → two-stage GBDT classifier →
constrained assignment with expected-F0.5 selection → ensemble and French post-processing.  
**Core Innovation:** (1) a from-scratch contrastive encoder over hashed character n-grams
that learns cross-script and noisy-name correspondences from the training pairs;
(2) competition, per-S1 "cluster agreement" and "introduced word / number" features that
exploit the one-S1-per-record structure and the look-alike generator; (3) cross-fitted training
data so encoder features behave on training pairs as on test; (4) a decision rule that optimises
the per-entity F0.5 metric directly.

---

## 3. Candidate Generation (Blocking)

Every record is normalised first (country-agnostic, so France needs no special handling):
`anyascii` transliteration of all scripts, mojibake repair, 0-for-o fixes, merging of dotted
initials (`S.A.S.` → `sas`), legal-suffix removal, state/region components pulled into a
separate field (US/India codes, native-script state names, French departments folded into
regions), and street/India/France abbreviation canonicalisation.

Candidates are retrieved per country, from the S2/S3 side (each record needs ≤1 S1):

| Retriever | Representation | Kept |
|---|---|---|
| Learned encoder | Two-tower model: hashed char 3-grams, words, house numbers, state codes → EmbeddingBag → MLP; symmetric InfoNCE with in-batch negatives grouped by (country, state) | top 5 (all 16 retrieved for address-less records) |
| SVD combined | Char-3-gram TF-IDF → TruncatedSVD, name ⊕ address | top 1 (4) |
| SVD name | Name half only | top 1 (4) |
| SVD address | Address half only | top 1 (4) |

Search is exact inner-product top-k on the GPU with a block-max pruning trick (take the
maximum of each 128-score block, then run top-k only inside the best blocks).

- **Blocking keys used:** learned embeddings, character n-gram TF-IDF (SVD) on name,
  address and both; country is a hard block.
- **Candidate pairs generated:** 66.0M per run for the test set (≈6.6 per S2/S3 record, ≈38
  per S1); the final ensemble scores the union of two runs built on different encoders,
  96.7M pairs (≈9.7 per record), out of 6.7×10¹² within-country pairs (reduction ratio
  > 0.99998). `candidate_pairs.tsv` is exactly this union.
- **How true matches were not lost:** the retrievers are complementary (name-only and
  address-only searches catch trade names and truncated addresses); records without an
  address keep every retrieved candidate. Candidate recall on the hold-out is **98.8%** of
  true pairs.

---

## 4. Matching Model

**Features used (66 in stage 1):**
- Name features: token-set/sort ratio, ratio, partial ratio, Jaro-Winkler, space-free
  ratio, token Jaccard, acronym match, legal-form equality/conflict, extra tokens and
  whether they are look-alike markers or positive-side noise words, name-key equality,
  name-key frequency among S1 and among S2/S3.
- Address features: ratio, token-set ratio, partial ratio, house-number overlap / Jaccard /
  conflict, state/region agreement, empty-address flags.
- Introduced words and numbers (v11): house numbers the record adds / drops, typo similarity
  between differing numbers (160 vs 169), whether the record introduces a genuine name word
  of its country (a word frequent in that country's S1 names that the variant generator never
  adds, and not a spelling of any replaced word), generator noise words added, look-alike
  marker added. A genuine introduced name word appears in 0.5% of true pairs vs 49% of
  candidate negatives. Vocabularies are counted per country on the split being scored.
- Retrieval and competition: encoder and SVD cosines, each retriever's rank, the pair's
  margin over the record's other candidates (by encoder cosine and by name similarity),
  number of near-identical names among the candidates, the S1's candidate count and how
  many records rank it first.
- Record flags: empty address, URL name, non-ASCII name, invented brand name (single word
  never used in any S1 name), short acronym name, source (S2/S3).
- Generator-recipe features (v12/v13, 13 more, all label-free and country-agnostic):
  canonical legal forms (corp = corporation, ltd = limited; SARL vs SA is a real change) and
  whether the record adds/drops one; the signed delta between the closest differing house numbers
  and between the first numbers (look-alikes use the S1's number +1..+11, real typos do not);
  for the words a record adds (drops), the per-country share of top-1 retrieval pairs with that
  change that keep an S1 house number / the same street (look-alike markers such as US "group"
  0.17, French "holding"/"international" 0.07 vs noise words "service" 0.67, "fils" 0.76);
  street-without-number equality. Stage-1 A/B on the hold-out: +0.0030 (v12) and +0.0005 (v13).
- Stage 2 adds the stage-1 probability and per-S1 cluster features: similarity of the record
  to the other records stage 1 links to the same S1 (best name/address match, shared house
  number, cluster size and probability mass).

**Model type:** two stages; stage 1 LightGBM, stage 2 LightGBM + CatBoost averaged (binary log loss, early stopping).
Encoder-derived features must behave on training pairs as they do on test, so training pairs
are **cross-fitted**: train S1 entities are split into folds, one encoder per fold is trained
without that fold's entities, and the fold's candidate pairs are scored with it. The final
matchers train on 22M such pairs (a learning curve showed ~+0.0014 macro F0.5 per doubling up
to ~22M; 30M rows added only +0.0001); a fixed 5% S1 hold-out, never used by any encoder or
matcher, measures everything. Stage 2 is trained on out-of-fold stage-1 probabilities.
**Model selection and tuning (v13):** on one S1-grouped hold-out slice (614k training rows,
694k test rows, macro F0.5 after the full decision rule) LightGBM 0.9811, CatBoost 0.9811,
XGBoost 0.9807, scikit-learn HistGradientBoosting 0.9808 and random forest 0.9742 (poorly
calibrated); averaging LightGBM and CatBoost gave 0.9815. An 11-setting LightGBM search and a
5-setting CatBoost search (`src/v13/hp_tune.py`, chosen by log loss) selected LightGBM with
feature_fraction 0.6 (otherwise 127 leaves, learning rate 0.05, min 100 rows per leaf, L2 1) and
CatBoost depth 10, learning rate 0.08, L2 3. The final stage 2 averages LightGBM 0.7 and CatBoost
0.3 (weight chosen on the hold-out: 0 → 0.98730, 0.3 → 0.98747, 0.5 → 0.9874, 0.7 → 0.9872).

**Unlabelled country (France):** the encoder is additionally trained on 796k pseudo-labelled
French pairs, the confident (p ≥ 0.98) assignments of an earlier run; this raised France's
median top-1 retrieval margin from 0.175 to 0.252 with US/India unchanged. After the decision,
a **French name-word veto** drops French pairs whose record introduces a genuine French name
word or a look-alike marker (label-free rule; `france_veto.py`). On the leaderboard the veto
alone moved 0.9756 → 0.9773 with only French rows changed. From v12 on, the recipe features
let the matcher reject these look-alikes itself (the veto would remove only 1.2% of French S1
matches, mostly genuine), so the final file does not apply it.

**Ensemble:** two complete runs built on different encoders (base and self-trained, big-data
matcher) are averaged with weights 0.1 / 0.9 over the union of their candidates (weight chosen
on the hold-out); countries without training labels use the self-trained run only.

**Threshold selection method:** each S2/S3 record is assigned to its highest-probability S1
if p ≥ τ (τ = 0.5); each S1 then keeps the prefix of its matches (sorted by p) that
maximises expected F0.5, EF(k) = 1.25·S_k / (0.25·T + k), against EF(0) = Π(1 − p), the
probability that the S1 is a singleton. τ and the rule were chosen by macro F0.5 on
out-of-fold hold-out predictions.

---

## 5. Results & Error Analysis

| Version | Hold-out macro F0.5 | Public leaderboard |
|---|---|---|
| Exact name + house-number rule | 0.655 | — |
| Learned retrieval + cosine rule + house-number agreement | 0.871 | — |
| Stage-1 LightGBM + expected-F0.5 selection | 0.977 | — |
| + address-less candidates, name-frequency and filler features | 0.979 | — |
| + stage-2 cluster matcher | 0.982 | 0.971 |
| + France self-training | 0.982 | 0.972 |
| + cross-fitted training data (22M pairs), weighted ensemble | 0.985 | 0.9756 |
| + French name-word veto | 0.985 | 0.9773 |
| + introduced word / number features (v11) | 0.9853 | 0.9784 |
| + legal-form / house-number-delta / added-word-share features (v12), no veto | 0.9870 | 0.9796 |
| + more recipe features, tuned LightGBM, CatBoost stage 2 (v13) | **0.9875** | **0.9815** |

- **F0.5 Score (macro):** 0.9875 on the hold-out (US 0.9885, India 0.9860; precision 0.9978,
  recall 0.9672).
- **Common false positives (wrong merges):** generated look-alikes that keep the address and
  house number but swap a name word or legal form, and address-less records linked to the
  wrong member of a same-name chain.
- **Common false negatives (missed matches):** records without an address whose name is shared
  by several S1 (chains; name-only evidence cannot pick the right one), trade names that share
  only an address with the S1, and heavily truncated addresses.
- **Unseen country:** a leave-one-country-out study (train on US, test on India) showed that
  zero-shot countries cost accuracy mainly through the encoder; France was the main source of
  the hold-out → leaderboard gap, addressed by self-training, French vocabulary handling and
  the veto.
- **Tried and rejected on the hold-out:** exact name + number recall booster (precision 4.9%,
  −0.016: the matcher had correctly rejected legal-form look-alikes), invented-name address
  booster (+0.00005), stricter singleton rules (all negative), probability recalibration
  (0), LightGBM capacity/learning-rate tuning (±0.0002), matcher training beyond 22M rows
  (+0.0001).

---

## 6. Conclusion

A learned character-level retriever plus competition-aware features turned a 44%-recall
rule baseline into a 0.9875 macro-F0.5 matcher (0.9815 on the public leaderboard). The biggest
gains came from modelling the data structure (one S1 per record, per-S1 clusters, what a
record introduces), optimising the metric directly, and giving the matcher honest,
cross-fitted training data at scale. Model capacity and probability calibration were not
bottlenecks. Generalisation to the unseen country was handled with country-agnostic
normalisation, self-training and a label-free French look-alike veto.

---

## 7. Use of Unlabelled Test Inputs

No test labels exist or are used, and no external data, services or pretrained models are
used. The pipeline does use the unlabelled test records in three label-free ways: the SVD
vectoriser is fitted on a sample of train + test text; the encoder is self-trained on the
model's own confident test predictions for France; and per-country word statistics (name
vocabulary, generator words) are counted on the split being scored, including test.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — `run_all.sh` runs everything in order; `README.md` lists
each stage with commands and timings.

| Script | Stage |
|---|---|
| `src/prepare_data.py` | TSV → parquet (RFC-4180 quoting, no NA coercion) |
| `src/normalize_data.py` | normalisation of every record (`er/normalize.py`) |
| `src/embed_data.py` | unsupervised char-TF-IDF SVD vectors (`er/vectorize.py`) |
| `src/featurize_data.py`, `src/train_encoder.py` | hashed features, encoder training (incl. `--pseudo`, `--exclude-fold`) + embeddings (`er/encoder.py`) |
| `src/retrieve.py` | GPU candidate generation (`er/knn.py`, `er/candidates.py`) |
| `src/build_features.py` | pairwise features (`er/features.py`), hold-out or cross-fitted fold rows |
| `src/train_matcher.py`, `src/train_stage2.py` | stage-1 / stage-2 LightGBM on hold-out rows (base run) |
| `src/train_big.py` | stage 1 + stage 2 on cross-fitted rows |
| `src/predict.py` | scores test candidates (`er/decide.py`, `er/cluster.py`) |
| `src/make_pseudo.py` | confident test assignments as pseudo-labels |
| `src/blend_weight.py`, `src/ensemble_predict.py` | hold-out choice of the blend weight; weighted ensemble over the candidate union |
| `src/france_veto.py` | French name-word / look-alike veto |
| `src/package_submission.py` | builds the submission zip |

### B. Additional Results

Blocking recall of the learned encoder on held-out S1 (R@8): US 0.986, India 0.983, versus
0.915 / 0.79–0.83 for the unsupervised SVD retriever alone.

Matcher learning curve, stage 1: on one hold-out fold, 0.7M training rows 0.9758, 1.4M 0.9772,
2.7M 0.9787; on the full hold-out with cross-fitted rows, 22M 0.9812 and 30M 0.9816.

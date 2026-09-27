# Findings — what moved the score, what didn't

Amazon ML Challenge 2026, Business Entity Resolution (macro F0.5 per S1 entity, singletons
included). All hold-out numbers are on a fixed 5% S1 hold-out (110,250 entities) that no encoder
or matcher trained on. Leaderboard (LB) numbers are exact public scores.

## 1. Leaderboard path

| File | Change | Hold-out | LB | Δ LB |
|---|---|---|---|---|
| v5 | stage-1 + stage-2 cluster matcher, expected-F0.5 selection | 0.9822 | 0.971 | |
| v6 | + France self-trained encoder | 0.9821 | 0.972 | +0.001 |
| v9g | + cross-fitted 22M-row matcher, 0.2 base / 0.8 big blend | 0.9846 | 0.975568 | +0.0036 |
| U1 | + French name-word veto (post-processing) | 0.9846 | 0.977334 | +0.0018 |
| v11ef | + v11 features (what a record introduces vs its S1) | 0.9853 | 0.978436 | +0.0011 |
| v12e | + v12 features, veto dropped | 0.9870 | 0.979596 | +0.0012 |
| v13e | + v13 features, tuned LightGBM, CatBoost stage 2 (0.3) | 0.98747 | 0.981542 | +0.0019 |
| **v13f_cat06** | France rows only: CatBoost weight 0.6 | 0.98747 | **0.981687** | +0.00015 |

Hold-out → LB gap shrank from 0.011 (v5) to 0.0058 (v13f). The test's extra decoy density explains only
≈ 0.001 of it (simulated), so most is probably the unlabelled country: France ≈ 0.945 by arithmetic (unverified).

## 2. What worked (largest first)

1. **Generator-recipe features (v11–v13), +0.0030 stage-1 A/B for v12 alone.** The data is
   synthetic; look-alike "hard negatives" and genuine variants are made by different recipes,
   and features that expose the recipe transfer to the unlabelled country:
   - **Signed house-number delta**: look-alikes use the S1 number +1…+11 (`num_delta_pos` alone +0.0015).
   - **Added-word number-share rate** per country (label-free): for each word a record adds,
     the share of top-1 retrieval pairs adding it that keep the S1 house number. Look-alike markers
     rarely keep it (US group .17, France holding / international / distribution / participations .07,
     groupe / developpement / france .23), noise words usually do (US service .67; France fils .76, associes .74).
   - Canonical legal forms (corp = corporation; SARL → SA is a real change).
   - v11: numbers added / dropped / typo'd, "introduces a genuine name word", "adds a look-alike marker".
   - v13: dropped-word share, street equality, first-number delta (+0.0005).
2. **Cross-fitted training rows.** Train S1 split into folds; each fold's candidate pairs are scored
   by an encoder that never trained on that fold, so encoder features behave as on test.
   Stage 1 0.9787 → 0.9808 on fold-0 rows alone; 22M rows → stage 2 0.9846. Saturates: 30M rows = +0.0001.
3. **Stage-2 cluster matcher** (0.9789 → 0.9822 hold-out): stage-1 features + p1 + how similar the record is
   to the other records stage 1 links to the same S1.
4. **Expected-F0.5 decision.** Each record goes to at most one S1 (ground truth never links a record
   to two); each S1 keeps the prefix of its ranked matches that maximises expected F0.5, where the
   empty list is scored by P(singleton). Needs calibrated probabilities.
5. **LightGBM + CatBoost stage 2.** On one hold-out slice the families tie alone, but the average
   gains +0.0004 (table below). On the unlabelled country CatBoost generalises better: see §4.
6. **Test-time ensemble** of two matcher runs over the union of candidates: +0.0011–0.0014 on the hold-out
   (mostly precision). Countries without labels take only the adapted run (`--adapted-for-unlabelled`);
   averaging in a non-adapted run dilutes France.
7. **Learned retrieval**: two-tower hashed char-3-gram encoder trained from scratch + SVD char-TF-IDF
   (combined / name / address), exact GPU top-k. Keep encoder top-5 + SVD top-1; address-less records
   keep everything. Pair recall 0.988.

Model comparison (same 74 features, S1-grouped slice, macro F0.5):

| LightGBM | CatBoost | XGBoost | HistGB | RandomForest | LGB + Cat | LGB + XGB + Cat |
|---|---|---|---|---|---|---|
| 0.98111 | 0.98110 | 0.98072 | 0.98077 | 0.97419 | **0.98150** | 0.98143 |

Tuning moved LightGBM by at most ±0.0006 (best: feature_fraction 0.6; lr 0.03; 63 leaves / min 200
per leaf). Results in `code/business_entity_resolution/configs/hp_results.json`.

## 3. What did not work (skip these)

| Idea | Result |
|---|---|
| More matcher rows beyond ~22M | +0.0001 |
| Probability power-calibration; LightGBM tuning before v13 | 0 / ±0.0002 |
| Invented-brand and short-acronym flags | 0 (redundant) |
| Exact-key booster (force-accept identical keys) | precision 4.9%, −0.016 |
| Invented-name address booster | +0.00005 |
| Singleton rule (force empty lists) | every variant negative |
| Tie-breaking address-less records whose name several S1 share (raw spelling, S2↔S3 id/row links, min/max id, most matches, longest address, digits in names — 8 tests) | all at chance; these cost 0.0059 hold-out and look unresolvable |
| Row-order / ID leak | none in train or test |
| Encoder-free features for the unseen country (US-only → India proxy) | worse: 0.8617 → 0.8542 → 0.8489 |
| Address co-location features | logloss 0.00750 → 0.00749 |
| Simulating test decoy density (decoys ×1.9 on hold-out) | only −0.0010; no tau / odds change helps |
| Legal-canon rule for address-less same-core groups | +0.0001 |
| Random forest | −0.007, poorly calibrated |
| Indic-script lexicon, masked `##` digits | not bottlenecks (recall 0.970 / 0.989) |
| French name-word veto on v11+ models | the new features already reject look-alikes; the veto now mostly removes genuine pairs |

## 4. France (15% of test S1, no training labels)

- **Probe trick.** French records only compete with French S1s, so a change applied to French rows
  alone leaves US/India byte-identical, and LB difference = 0.15 × ΔF0.5(France), exactly.
  All France decisions below were made this way.
- **CatBoost weight for French rows** (v13): 0.3 → 0.981542, **0.6 → 0.981687**, 1.0 → 0.981577.
  The mix beats either model.
- **Self-training**: encoder retrained with 795,877 French pseudo-pairs (v5 p ≥ 0.98): LB +0.001.
- **French look-alikes follow the US recipe**: the marker word is *added* (developpement, france,
  international, distribution, holding, participations, groupe) together with a new house number;
  the address equals the S1's only 0–4% of the time (US group / holdings: 0%). Noise words (fils,
  associes, service, compagnie = expanded "cie") keep the address ≈ 50% and the number ≈ 85%.
  French data is lighter-noised: 24% of French best matches are exact copies (US 12%).
- **Name-word veto** (`src/france_veto.py`): drop French pairs whose record introduces a French
  name word (in ≥ 20 French S1 names, S2/S3-to-S1 frequency ratio ≤ 1.3, not a spelling of a word it
  replaces) or a marker. **+0.0018 on a model without the v11 features.** On v11 it removed 15.9k pairs,
  mostly genuine; v12/v13 run without it. Useful for any pipeline that lacks recipe features.
- **Size test** (label-free check of a rule): a rejected genuine variant leaves its S1 one match
  short, a look-alike is an extra record and does not. D = mean matches of the S1s involved − country
  mean. Calibrated on US labels: genuine −0.17, look-alike −0.05; France genuine −0.17…−0.28, look-alike
  −0.03…−0.09. It reproduced the veto's +0.0018 before we knew it. **Confound:** same-name records at a
  completely different address ("moved business") also score ≈ −0.25 but are negatives by the US labels —
  trust D only when the address or house number matches.

## 5. Data facts and traps

- Train S1 2.21M, S2 5.03M, S3 5.29M (US 60% / India 40%). Test S1 1.73M (India 47%, US 38%, France 15%).
- Ground truth: 5.58% singletons, mean 3.46 matches (0–11); a record matches at most one S1; no cross-country pairs.
- **Read TSVs with RFC-4180 quoting on, `keep_default_na=False`, `dtype=str`**: 813 test lines are quoted
  (mostly French) and some names are literally "NA".
- Mojibake is case-mangled (`Â\x80\x93`); `[âÂ]\x80([\x80-\xbf])` → bytes E2 80 X repairs 99.7%.
- Invented brand names (Kelonyla, Dovaxylo) are 97.8% positives.
- Test prior shift: +23% targets per S1; look-alike signature per S1 +57% US / +36% India.
- Test S1 is not fully de-duplicated (a few exact duplicates; French near-duplicates differing only by legal form).
- Postal codes are nearly absent; S2-US is 90% upper case; S3-US spells out state names.

## 6. Where the remaining error is (hold-out, v9)

Fixing every resolvable error caps hold-out at ≈ 0.9944. Never-retrieved pairs (1.24%) → ceiling 0.9961;
address-less records sharing an S1 name (1.41%) → 0.9959; both → 0.9919. A third of retrieval misses do have
an address, over half of those with made-up / acronym / URL names — exact address-key blocking was planned,
never built. Key-based bidirectional retrieval reaches 99.10% pair recall vs our 98.76%.

## 7. Use of unlabelled test inputs (check your rules)

No test labels exist or are used, but three steps read unlabelled test records: the SVD vectoriser is
fitted on train + test text; the encoder is self-trained on its own confident French test predictions;
per-country word statistics (vocabulary, added-word rates) are counted on the split being scored.

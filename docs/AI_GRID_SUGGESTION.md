# Using AI to design the XGBoost hyper-parameter grid

## Honest provenance
* **Space v1** (`SEARCH_SPACES["XGBoost"]` in `src/models.py`) was drafted with AI assistance while the project was built.
* **Space v2** (`AI_SUGGESTED_XGB_SPACE`) was obtained by giving the AI assistant (Claude) the *fresh, explicit
  prompt below* and writing its answer down. Because both came from AI, this is a comparison of two AI-designed
  spaces; run `python -m src.ai_grid_experiment` to reproduce it.
* Tip: paste the same prompt into ChatGPT/Gemini/Claude yourself and add its answer at the bottom of this file
  to compare a third opinion.

## The prompt

```
I am training an XGBClassifier (scikit-learn Pipeline, RandomizedSearchCV, 5-fold stratified CV,
primary metric ROC-AUC) on a binary customer-churn dataset: about 5,600 training rows,
~45 features after one-hot encoding, 26.5% positive class. I have 1 CPU core and a budget
of 40 random-search candidates (about 1 minute).

1. Suggest a search space using scipy.stats distributions (randint, uniform, loguniform) for the
   parameters that matter: n_estimators, learning_rate, max_depth, min_child_weight, subsample,
   colsample_bytree, gamma, reg_alpha, reg_lambda, scale_pos_weight.
2. Explain why each range suits THIS data size (not generic ranges).
3. Rank the parameters by how much they usually matter.
4. Tell me which interactions to respect (for example learning_rate vs n_estimators).
```

## The AI's answer (summary)

| Parameter | Suggested distribution | Reasoning given |
|---|---|---|
| `n_estimators` | `randint(200, 800)` | pairs with a small learning rate; more trees needed when each step is small |
| `learning_rate` | `loguniform(0.01, 0.2)` | log scale; above ~0.2 overfits quickly on 5.6k rows |
| `max_depth` | `randint(3, 7)` (3..6) | small tabular data rarely needs deep trees; depth is the main complexity control |
| `min_child_weight` | `randint(1, 8)` | larger values stop splits on tiny groups |
| `subsample` | `uniform(0.7, 0.3)` (0.7..1.0) | row sampling for variance reduction, but below 0.7 is too noisy on small data |
| `colsample_bytree` | `uniform(0.6, 0.4)` (0.6..1.0) | feature sampling; 45 one-hot features is fairly redundant |
| `gamma` | `uniform(0, 3)` | minimum gain to split |
| `reg_alpha` | `loguniform(1e-3, 5)` | L1; mild effect |
| `reg_lambda` | `loguniform(0.5, 20)` | L2; stronger than the default of 1 usually helps here |
| `scale_pos_weight` | `[1, 1.66, 2.77]` | none / sqrt(neg:pos) / full neg:pos ratio; full weighting can hurt probability quality |

**Importance ranking given:** (1) learning_rate + n_estimators, (2) max_depth, (3) min_child_weight,
(4) subsample and colsample_bytree, (5) reg_lambda / reg_alpha / gamma, (6) scale_pos_weight
(matters for F1/recall, much less for ROC-AUC).
**Interactions to respect:** lower learning_rate needs higher n_estimators; deeper trees need more regularisation.

## Differences from space v1

| Parameter | v1 (original) | v2 (AI, fresh prompt) |
|---|---|---|
| `max_depth` | 2..7 | **3..6** (narrower) |
| `n_estimators` | 100..600 | **200..800** (higher floor) |
| `learning_rate` | 0.01..0.3 | **0.01..0.2** (lower cap) |
| `subsample` / `colsample_bytree` | 0.6..1.0 / 0.5..1.0 | **0.7..1.0 / 0.6..1.0** |
| `reg_lambda` | 0.01..10 | **0.5..20** (stronger L2 floor and ceiling) |
| `gamma` | 0..5 | **0..3** |
| `scale_pos_weight` | {1, 2.77} | **{1, 1.66, 2.77}** (adds a middle option) |

The idea behind v2: a **narrower, better-informed space** wastes fewer of the 40 trials on bad regions.
Whether that actually helps is tested empirically in `src/ai_grid_experiment.py` (results in the README).

> Do not trust AI-suggested ranges blindly: treat them as a hypothesis and verify them with the search results.

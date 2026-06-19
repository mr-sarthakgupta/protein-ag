# Loss Function Investigation: `disease_relevant_noninhibited_all`

Generated from archived run: `/home/mrsar/protein-ag/skydiscover/outputs/adaevolve/disease_relevant_noninhibited_all_0617_0742`  
Best archived candidate: stretched-exponential/JMAK template in `best/best_program.py`.

## Executive summary

The current objective is not just “simple NMSE”. It is per-dataset NMSE, normalized by each full dataset's response variance, then averaged with one weight per validation point. For the best archived candidate, this gives weighted validation NMSE `0.05311` and combined score about `0.9199` after parsimony. The same candidate has unweighted per-dataset mean NMSE `0.22488` and median `0.02214`, so the reported scalar is substantially shaped by dataset size.

Most concerning findings:

1. **Point weighting dominates the objective.** The top 3 datasets by aggregate loss contribution account for `74.2%` of the final NMSE. The top 5 largest validation datasets account for `92.9%` of all validation points before error is even considered.
2. **The metric is not a clean generalization score.** `12` evaluation-only datasets fit constants on the exact points they score; `17` loaded datasets with no validation points are excluded from the aggregate.
3. **MSE pressure is region-dependent.** Regions with more sampled points and larger absolute residuals dominate the squared-error objective, while small-amplitude or transition-local shape errors can be visually important but weakly represented by a global dataset variance denominator.
4. **The inner optimizer and outer score are misaligned.** Constants are fit by unnormalized residual MSE/soft-L1 on the fit split, but candidates are selected by variance-normalized validation MSE aggregated across datasets.

## Saved plots

- `loss_weight_concentration.svg`: dataset point-share vs final loss-share.
- `loss_region_diagnostics.svg`: normalized time and normalized response bins, comparing sample density against squared-error share.
- `candidate_curve_fit_examples.svg`: highest-impact validation curve examples from the archived best candidate.
- `adaevolve_loss_score_history.svg`: run score history showing when the best candidate emerged and where later children plateaued.

## Dataset weighting

| Dataset | validation points | NMSE | point share | aggregate loss share | evaluation-only |
|---|---:|---:|---:|---:|---|
| `Abeta/Meisl2014/Ab40_sec_IN_HOURS` | 4796 | 0.0469 | 35.9% | 31.7% | yes |
| `Abeta/Weiffert2019_for_seeded/Ab42_in_seconds` | 209 | 0.8628 | 1.6% | 25.4% | no |
| `insulin/Nielsen2001_seeded_sec` | 47 | 2.5699 | 0.4% | 17.0% | no |
| `Alphasyn_Gaspar2017/03uM_seed all` | 517 | 0.1445 | 3.9% | 10.5% | no |
| `biofilm proteins/CsgA/CsgA_IN_HOURS` | 3780 | 0.0097 | 28.3% | 5.2% | yes |
| `Abeta/Cohen2013/Ab42_sec_IN_HOURS` | 2542 | 0.0096 | 19.0% | 3.4% | yes |
| `gelsolin/Fig4C` | 20 | 0.8143 | 0.1% | 2.3% | no |
| `htt/Kakkar2016/Kakkar2016_6kDa_polyQ_with_seed_sec` | 144 | 0.1058 | 1.1% | 2.1% | no |

Because final NMSE is `sum(dataset_nmse * n_val_points) / sum(n_val_points)`, adding points to one dataset changes the scientific objective even if the number of biological systems is unchanged. This encourages the search to improve high-sample datasets first, not necessarily the most mechanistically revealing datasets.

## Region pressure

Time-bin diagnostics:

| normalized time bin | points | point share | squared-error share | MSE |
|---|---:|---:|---:|---:|
| 0.0-0.2 | 4912 | 36.8% | 42.1% | 0.0077 |
| 0.2-0.4 | 3529 | 26.4% | 28.9% | 0.0073 |
| 0.4-0.6 | 2200 | 16.5% | 16.8% | 0.0068 |
| 0.6-0.8 | 1570 | 11.7% | 11.5% | 0.0066 |
| 0.8-1.0 | 1152 | 8.6% | 0.7% | 0.0005 |

Response-bin diagnostics:

| normalized y bin | points | point share | squared-error share | MSE |
|---|---:|---:|---:|---:|
| 0.0-0.2 | 4436 | 33.2% | 36.5% | 0.0074 |
| 0.2-0.4 | 1068 | 8.0% | 21.6% | 0.0181 |
| 0.4-0.6 | 1075 | 8.0% | 17.8% | 0.0148 |
| 0.6-0.8 | 1582 | 11.8% | 14.5% | 0.0082 |
| 0.8-1.0 | 5202 | 38.9% | 9.6% | 0.0016 |

This matters for protein aggregation because the lag phase, transition steepness, and plateau can have different mechanistic meaning. A single pointwise MSE can prefer a curve that is numerically close over many plateau points while missing onset timing, or can overreact to a few steep-transition residuals depending on sampling density.

## Worst individual validation curves

| Curve | points | MSE | MAE | max abs error | y range | evaluation-only |
|---|---:|---:|---:|---:|---:|---|
| `insulin/Nielsen2001_seeded_sec` curve 2 | 47 | 0.3810 | 0.5134 | 0.9566 | 1.000 | no |
| `Abeta/Weiffert2019_for_seeded/Ab42_in_seconds` curve 3 | 209 | 0.1149 | 0.2388 | 0.7520 | 0.945 | no |
| `gelsolin/Fig4C` curve 2 | 20 | 0.1100 | 0.2733 | 0.5542 | 0.890 | no |
| `Alphasyn_Gaspar2017/03uM_seed all` curve 23 | 58 | 0.0253 | 0.1481 | 0.2262 | 0.975 | no |
| `Alphasyn_Gaspar2017/03uM_seed all` curve 19 | 58 | 0.0247 | 0.1452 | 0.2402 | 0.981 | no |
| `serum amyloid/Ye2011_sec` curve 3 | 8 | 0.0237 | 0.0916 | 0.3602 | 0.966 | no |
| `Alphasyn_Gaspar2017/03uM_seed all` curve 17 | 56 | 0.0227 | 0.1287 | 0.3032 | 0.976 | no |
| `htt/Kar2011/Kar2011_Q37_sec` curve 0 | 8 | 0.0218 | 0.1080 | 0.2940 | 0.996 | no |
| `Alphasyn_Gaspar2017/03uM_seed all` curve 18 | 58 | 0.0193 | 0.1273 | 0.2370 | 0.955 | no |
| `Abeta/Meisl2014/Ab40_sec_IN_HOURS` curve 0 | 261 | 0.0181 | 0.0850 | 0.3536 | 0.870 | yes |

No validation curves in this split had `y_range < 0.25`; high-amplitude curves (`y_range >= 0.75`) had mean curve MSE `0.01420` across `69` curves. This still supports checking curve-level normalization or shape metrics, because global per-dataset min-max/variance normalization can hide within-dataset curve amplitude differences when such curves are present.

## Potential objective changes

1. **Dataset-balanced objective:** average NMSE equally across scoring datasets, or cap each dataset's contribution. This prevents large validation files from dominating.
2. **Curve-balanced objective:** compute per-curve losses first, then average curves within each dataset. This better matches the experimental unit: each X/Y trajectory.
3. **Shape-aware terms:** add small penalties for onset/half-time error, slope/derivative mismatch, or monotonic/saturating violations. These capture governing-law behavior that pointwise NMSE can miss.
4. **Robust loss for selection:** use MAE/Huber/log-cosh or quantile-trimmed MSE at the outer aggregation level to reduce overreaction to isolated residual spikes.
5. **Separate validation semantics:** do not mix evaluation-only fit-and-score datasets with held-out-curve validation in the same scalar, or report them as two objectives.
6. **Align inner and outer losses:** fit constants against the same normalized/weighted residuals used in model selection, otherwise the optimizer may choose constants that are locally good for unnormalized training MSE but suboptimal for the scored criterion.

## Suggested next experiment

Keep the current evaluator as a baseline, then compare candidate rankings under three alternative aggregates using the archived checkpoint programs: point-weighted NMSE, dataset-balanced NMSE, and curve-balanced robust Huber loss. If the top programs reorder materially, the current loss is steering discovery rather than merely measuring fit.

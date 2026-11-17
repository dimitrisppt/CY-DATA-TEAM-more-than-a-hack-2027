# Possession experiments and decisions

This document summarizes the measured development path to v6. Scores are
whole-match spell F1 aggregates unless stated otherwise. Values from different
folds, masks, seeds, and objectives are not interchangeable. None is a
challenge-match score.

The fresh CUDA reconstruction of the complete possession-refinement comparison
is documented in [Comprehensive GPU IoU study](GPU_IOU_EXPERIMENT.md). It rebuilds
the missing historical parents and evaluates all three folds, two seeds and
three continuation variants, with a temporal-IoU sweep from 0.1 through 0.9.
All 54 GPU fits completed. On the historical refinement masks, spell balancing
scored 0.611849 against 0.604216 for the matched control and 0.596663 for the
fresh GPU baseline. It passed the development screen under both assessment
mask sets; masked-teacher consistency failed against control. Spell balancing
also increased short-possession recall and spurious events. That screen
advanced it to fresh-seed confirmation, with no change to the delivered v6 model.

The subsequent 80-fit GPU follow-up completed on 17 November. The frozen
combined recipe passed fresh-seed confirmation (+0.002796 versus matched
control, five of six match wins) and the separate WMX guard (+0.020419).
The gain depends on the two-seed ensemble: both individual fine-seed
comparisons declined in the six-match repeat. Sampling-only and loss-only
ablations gained +0.002493 and +0.003172 using the original parents, but each
won only three of six matches and failed the breadth gate. The combined
recipe remains the strongest tested condition, with a persistent increase in
spurious short/wrong-team possessions. The delivered v6 model is unchanged;
new fully labeled matches are the next validation priority. Details and
artifact paths are in the follow-up section of the GPU study document.

The subsequent [eight-hour holistic GPU campaign](GPU_TUNING.md) finished on
17 November after 7 hours 50 minutes. It completed 23 screening recipes,
5 full-budget recipes and 540 decoder trials. The frozen `trial_0017` addition
and selected decoder improved the six-match weighted confirmation objective
from 0.604912 to 0.629368 and the WMX guard from 0.557923 to 0.569078.
Both confirmation seeds and all three additional fixed-recipe seed checks
improved. These remain development results on historically reused matches.
The winning pipeline has not yet been packaged for challenge inference;
the delivered v6 model and submission CSV are unchanged.

## Chronology

| Date | Phase | Scope and method | Result and decision |
|---|---|---|---|
| 26 Oct | Initial model audit | Seven leave-one-match-out evaluations. Each outer training set used three whole-match folds. Multivariate Optuna TPE compared mean pooling and ball-conditioned attention at the available trial budgets, then tuned a geometric-reference blend and decoder on out-of-fold predictions. | Mean-pooling models blended with the geometric reference were selected in all seven outer folds. Mean spell F1 was 0.4701 clean and 0.3734 under 41% synthetic ball outages, versus 0.3271 and 0.2771 for the fixed reference. These are historical development results. |
| 26 Oct | Augmentation search | Eleven recipes, including geometry, coordinate noise, occlusion, temporal transforms, sensor loss and combinations. Three match-disjoint folds covered six matches. Fits used 1,200 updates and paired seeds. A fresh-seed repeat and a separate excluded-match check followed selection. | Coordinate noise won the small search but regressed in the fresh-seed repeat and on average over the excluded-match seeds. It was not promoted as a general default. Later selected mean models use the specific coordinate-noise policy from the separate overnight search. |
| 26 Oct | Training stability and temporal policy | Three seeds across three whole-match folds were evaluated at 1,200, 2,400 and 3,600 updates. A separate decoder search compared no temporal transform, warp and shift with a fixed 16-trial decoder search per model. | The tested objective rose through 3,600 updates, so 3,600 was the best tested budget, not a demonstrated plateau. Time shift passed its initial mean gate, but a fresh seed improved short-gap F1 on only three of six matches. The declared four-match gate failed. Interaction models regressed and visibility-dependent blending matched the static blend. None replaced the later selected recipe. |
| 28 Oct | Overnight joint campaign | A six-match development set used three whole-match folds, staged model budgets of 1,200, 3,600 and 7,200 updates, 36 initial model configurations, and 976 recorded decoder evaluations. Conditional grouped multivariate TPE searched training, augmentation, reference, blend and decoder settings. Confirmation used two fresh training seeds, with separate WMX checks at two seeds. | The selected mean-pooling and ball-attention pair improved the weighted objective from 0.51095 to 0.56660 on that campaign's confirmation conditions. The pair became the frozen v2 recipe. These values use different masks and scope from the later v4-v6 table. |
| 16 Nov | Joint decoder polish | A fixed 64-trial search reused out-of-fold predictions from the selected pair. It varied pair and neural-reference weights, calibration, boundary contribution and typed transition costs. The best candidate was tested on fresh masks and paired seeds. | Search OOF objective rose by 0.005343, but fresh-mask confirmation gained only 0.000259 on average. One seed declined, so the two-seed gate failed. The decoder-only change was rejected. Boundary strengths above 2 also reduced the search objective. |
| 16 Nov | Seed averaging confirmation | The mean and attention architectures were averaged across two training seeds, using the frozen refined blend and decoder, then tested on two new outage-mask batches. Each batch retained three inner folds and a WMX guard. No further tuning occurred. | The combined four-network recipe passed both batch gates. Its objective was 0.580751 versus 0.565863 for the original individual-seed baselines, with positive missing-ball gains on all six inner matches. Since the recipes also differ, this supports the complete v4 predictor rather than isolating the effect of seed averaging. |
| 16 Nov | Fine-model tail EMA | Each selected fine model resumed at update 7,000 and replayed the final 200 updates with decay 0.99. Exact raw model and training-state parity were required before publishing averaged weights. Both frozen outage batches and WMX guards were scored. | The four-network EMA candidate passed its gates. Its matched objective was 0.590127 and missing-ball aggregate 0.525637, versus 0.580751 and 0.512152 for v4. EMA is over the final 200 updates only. |
| 16 Nov | Player-only coarse context | Two one-hertz seeds per split produced eight validation fits, each trained from scratch for 7,200 updates. Ball input was always hidden. Their interpolated outputs were gated into the fine live and owner probabilities, with no decoder or weight search. The frozen EMA candidate was the baseline. | Both outage batches and WMX guards passed. Objective rose from 0.590127 to 0.594200 and missing-ball aggregate from 0.525637 to 0.533782. Clean arrays were unchanged. This supports the added model capacity and context as a package, not a causal claim that temporal context alone produced the gain. Two separate final models were then fitted on all seven matches. |

## Search and validation choices

The main overnight campaign used conditional grouped multivariate TPE for
interacting architecture, training, augmentation, reference, blend and decoder
parameters. It allocated the same initial recipe quota to each architecture,
preserved unaugmented anchors, and promoted complete recipe rounds through
longer training budgets. Decoder selection used cross-fitted predictions.
Confirmation froze a single recipe before evaluating new training seeds and
outage masks. The official whole-match scorer and equal-match averaging were
retained.

The later joint decoder polish used 64 deterministic trials, including 22
anchors, with fixed proposal seed and resumable trial records. Its
confirmation failed, so it did not replace the original two-network recipe.
The refined blend and decoder were later included in a separately validated
four-network candidate. The later bagging, EMA, and coarse studies were
fixed-candidate tests. They reused three
inner folds, seeds 8101 and 8102, and separate WMX outer seeds 9101 and 9102.
Short and long synthetic gap masks changed by batch. Clean and visibility
replay conditions were reused. New masks measure sensitivity to corruption,
not unseen-match generalization.

Earlier experiments also tested coordinate noise, player and region occlusion,
time warp and shift, sensor loss, hold and blur, and combinations. Coordinate
noise was not consistently beneficial across repetitions. Time shift had a
positive mean in its initial study but failed the predeclared breadth gate on
fresh seed 13. Player interactions regressed on all six matches in that
comparison. A quality-aware blend selected equal visible and missing weights,
which was the static blend. These outcomes informed later fixed choices and
were not silently folded into v6.

## Interpretation and next questions

All seven public matches have prior development use. Outer-match labels and
fresh random seeds do not restore independence after repeated model selection.
The v4-v6 table records matched development comparisons for successive fixed
changes. It is not an estimate of challenge leaderboard performance.

The coarse branch adds parameters and compute as well as longer temporal
context. A same-capacity fine-only control has not been run. One-hertz sampling
may miss brief events and high-frequency player motion. The later GPU study
tests longer fine training, complete-possession balancing, and masked-teacher
consistency. A fold-safe sequence model, improved handling of very long gaps,
and evaluation on additional compatible labeled matches remain untested
hypotheses. They should be tested with frozen folds, seeds and promotion rules
before changing the delivered recipe.

The frozen v4-v6 per-match scores and decoded labels are in
`benchmarks/v6/public-reference/`. The model bundle contains authenticated
training and validation proofs under `models/v6/`. Use the commands in
[reproduction details](REPRODUCIBILITY.md) to verify the model and recalculate
the scores. Earlier experiment implementations remain in `experiments/`,
including their match splits, search spaces and promotion gates.

An earlier SkillCorner transfer diagnostic covered 47,121 of 118,494 frames
across two matches. Its labels were provider-generated home/away agreement
on a conditional subset, with no dead-play state. It did not train or select
v6, and cannot establish full three-state possession accuracy.

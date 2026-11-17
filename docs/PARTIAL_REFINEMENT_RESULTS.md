# Findings from the stopped refinement study

Possession-balanced training is promising on the completed validation fold.
The teacher objective adds essentially no measured benefit beyond its matched
extra-training control. The evidence covers only WOH and WOY, two historically
reused public matches. The other two folds were not trained.

Training remains stopped. This analysis decodes and scores saved predictions.
It performs no training, new model inference, tuning, promotion or submission
change. The original frozen study protocol and incomplete result state remain
intact. This partial analysis was requested after stopping and is descriptive.

## Comparison using both fine-model seeds

Both fine architectures finished for both seeds in the control and
possession-balanced variants. Their four-model ensemble retains the original
two coarse models, masks, reference blend and decoder. The reconstructed
baseline predictions exactly match frozen v6 on every included half and
scenario.

| Method | Weighted spell F1 | Delta vs v6 | Missing-ball aggregate | Clean spell F1 |
|---|---:|---:|---:|---:|
| v6 baseline | 0.612521 | 0 | 0.549127 | 0.675915 |
| Extra-training control | 0.611245 | -0.001276 | 0.552366 | 0.670123 |
| Possession-balanced training | 0.618443 | +0.005922 | 0.557981 | 0.678904 |

Possession balancing improves over the matched control by 0.007198. It improves
the mean clean, short-gap, long-gap and replay scores, and the weighted objective
in both mask batches. The match-level benefit is uneven:

| Match | v6 | Control | Possession balanced | Balanced delta vs v6 |
|---|---:|---:|---:|---:|
| WOH | 0.625893 | 0.618222 | 0.618872 | -0.007022 |
| WOY | 0.599148 | 0.604268 | 0.618014 | +0.018866 |

In the clean scenario, at temporal IoU 0.5, possessions shorter than two seconds
improve from 71/238 matched (29.8%) to 83/238 (34.9%). This recovers 12 additional
short possessions. Spurious possessions increase from 141 to 171 across all
durations. Better short-event recall therefore comes with more false events.

These two-match scores must not be compared directly with the earlier six-match
v6 objective of 0.594200.

## Teacher comparison using the common completed fine-model seed

Only seed 8101 completed both fine architectures for all three variants. The
teacher comparison therefore uses that seed for every method, including its
baseline. Both original coarse seeds remain fixed for every method. This is
a different predictor from the full four-fine-model v6 ensemble above.

| Method | Weighted spell F1 | Delta vs extra-training control |
|---|---:|---:|
| Matching baseline | 0.586964 | -0.007842 |
| Extra-training control | 0.594806 | 0 |
| Possession-balanced training | 0.599825 | +0.005018 |
| Masked-teacher supervision | 0.594668 | -0.000139 |

Teacher supervision improves over its original baseline, but the same extra
training already achieves that gain. Its missing-ball aggregate is only
0.000452 higher than the matched control. One match improves while the other
worsens. This provides no convincing incremental benefit for the teacher
objective at the tested settings. It does not establish that the broader
method cannot help.

Possession balancing also beats the control in this common-seed comparison,
with gains on both matches. This is consistent with the two-seed average,
but both comparisons reuse the same two matches.

## Training and integrity findings

Eleven fits finished all 1,200 extra updates and exported 200-update tail EMAs.
The twelfth fit stopped at saved update 7,800, after 600 extra updates, before
its EMA phase. It is excluded from all comparisons.

An independent artifact audit passed 154 checks. Exported weights match their
saved EMAs, all checked values are finite, and actual training-window
transcripts contain only WN1, WPY, WQQ and WR9. WOH and WOY are excluded.
Completed teacher runs share the control's training-window hashes and
augmentation counts. Short-spell sampling selected 278/1,200 windows for seed
8101 and 329/1,200 for seed 8102, with no unavailable draws.

Training loss cannot rank these methods. Recorded base-loss traces omit the
auxiliary objective and include inherited training history. Sampling also
changes the difficulty of training examples. The scores above come from
assessment predictions, not training-loss comparisons.

## Decision and reproduction

Retain the delivered v6 model. Possession balancing is the strongest candidate
for a future completed validation study. The teacher objective has no observed
incremental benefit here. No formal advancement gate can be evaluated from
only two of the six planned assessment matches.

Recompute both comparisons from the committed public label reference, without T7,
tracking data or models:

```bash
uv run --no-sync python -m scripts.evaluate_partial_reference
```

The command verifies the label archive, every label-array hash and the saved
result hash. It runs the official scorer for both groups and every variant,
match, scenario and mask batch. It also reproduces every saved batch 1
short-spell count and checks the full result against the saved expected values.
Halves are separated by an excluded frame when computing whole-match scores.
The objective averages the two mask batches within each match, then gives the
two matches equal weight. It uses 0.50 clean, 0.20 short-gap, 0.15 long-gap and
0.15 replay spell F1. The missing-ball aggregate is the last three weighted
terms divided by 0.50. Seeds and masks are not independent matches.

Inspect a descriptive temporal-IoU sweep from 0.1 to 0.9, with a CSV and a
readable report, using:

```bash
uv run --no-sync python -m scripts.analyze_partial_iou
```

The official objective averages IoUs 0.3, 0.4, 0.5, 0.6 and 0.7. The sweep
checks that it reproduces this objective and preserves both matched-seed
comparisons. Add `--prepared-dir data/processed/dfl` to verify that freshly
prepared public DFL truth exactly matches all four saved assessment halves.
Outputs default to `outputs/possession-refinement/iou-analysis/`. This analysis
does not resume training or evaluate the complete-study advancement gate.

The published reference contains only public DFL ground truth, decoded labels
and compact scores. It includes no participant inputs, target labels, neural
outputs, checkpoints or raw cache metadata. Its two-seed baseline and ground
truth arrays match the existing [v6 public reference](../benchmarks/v6/public-reference/manifest.json)
exactly. The [third-party notices](../THIRD_PARTY_NOTICES.md) document the public
DFL source and attribution. The historical training audit described above
requires the original local study storage. The portable command checks
label integrity and scoring, not the unpublished training states or audit logs.

The source result SHA256 is
`1392ae6b37519fff698674d2b25a63eeb43b1a6c371a757ee8c5a8183d238b32`.
The original protocol SHA256 is
`898265957fad1e9983324d55c62f65c09fce03f1401654993681e612ecace378`.
These identify the frozen pre-publication records. The manifest also retains
the original analysis script and source manifest SHA256 values.

See the [compact results](../benchmarks/partial-refinement-v1/result.json),
[manifest](../benchmarks/partial-refinement-v1/manifest.json) and
[decoded labels](../benchmarks/partial-refinement-v1/labels.npz).
The full [study design](SHORT_POSSESSION_REFINEMENT.md) remains available.
This is one completed assessment fold, two historically reused matches and
11 of 36 planned fits. Training remains stopped, and no promotion gate has
been evaluated.

# Comprehensive GPU IoU study

**Current release:** the confirmed combined refinement is included in v7, together
with the subsequent GPU tuning winner. See [GPU tuning](GPU_TUNING.md) and
[reproduction](REPRODUCIBILITY.md). References below to unchanged v6 describe
the historical experiment stages.

This is a fresh reconstruction of the possession-refinement experiment. The
historical T7 training states are unavailable, so its 11 completed CPU fits are
not mixed into the new comparison. Every parent, teacher, and coarse model is
rebuilt on CUDA using the published v6 recipes and original fold seed mapping.

## Completed results — 16 November 2026

All 54 fits completed on a Tesla T4: 172,800 optimizer updates in about
69.8 minutes with three fold workers. Both assessments completed on six
matches, covering 856,677 labeled native-rate frames across 12 halves.

The table gives the ensemble's official weighted possession F1, averaged over
temporal IoUs 0.3–0.7. The two mask sets use the same models and matches and
must not be pooled as independent evidence.

| Method | Historical refinement masks | Additional campaign masks |
|---|---:|---:|
| Fresh GPU baseline | 0.596663 | 0.597421 |
| Matched continuation control | 0.604216 | 0.603953 |
| Complete-possession-balanced continuation | **0.611849** | **0.612883** |
| Masked-teacher continuation | 0.602562 | 0.601445 |

Spell balancing passes the frozen development screen under both mask sets.
On the historical masks it gains **0.007633** over the matched control, with
a descriptive paired-match 95% bootstrap interval of **[0.001883, 0.014484]**.
It improves five of six matches, both mask batches, and all four scenario
averages. WOH declines by 0.001976. The gain varies across fine seeds:
0.002839 for seed 8101 and 0.016537 for seed 8102. Both coarse seeds remain
fixed in these individual-fine-seed comparisons.

The additional mask set gives a 0.008930 gain over control, with a descriptive
interval of [0.002692, 0.016190]. Its long-gap scenario declines slightly
(-0.001070), so the benefit is not uniform across every condition. The
masked-teacher continuation underperforms the matched control on both sets
and fails both development screens.

There is a short-possession tradeoff. On the shared clean inputs at IoU 0.5,
spell balancing detects 232/728 possessions shorter than two seconds (31.9%),
versus 196/728 (26.9%) for control. All-duration spurious possessions increase
from 457 to 533. These are pooled event counts across the six matches, using
one clean batch to avoid duplicate counting.

Spell balancing is a candidate for fresh confirmation and the separate WMX
guard. No delivered model or submission was replaced. All assessment matches
have prior development use; these results do not establish unseen-match or
challenge performance.

The run passed 134 targeted tests, including actual CUDA resume/inference
checks. All 3,072 prediction records passed checkpoint/input/device identity
and file-hash verification. There are 2,112 distinct prediction files after
960 exact-input cache reuses. Both complete score sets reproduced exactly
from their authenticated decoded-label references. Verification receipts are
`implementation-verification.json`, `prediction-integrity.json`, and
`completion-verification.json` under the study output directory.

Detailed artifacts:

- [Historical-mask report](../artifacts/gpu-iou-v1/historical-masks/report/report.md)
- [Additional-mask report](../artifacts/gpu-iou-v1/report/report.md)
- [Historical-mask IoU figure](../artifacts/gpu-iou-v1/historical-masks/report/iou-curves.png)
- [Historical-mask full results](../artifacts/gpu-iou-v1/historical-masks/report/result.json)

## Fixed design

The study evaluates six public DFL development matches in three whole-match
folds. Each fit trains on four matches and excludes both assessment matches.
Both halves stay together. WMX remains outside this screen. All seven public
matches have historical development use; this is not an independent test.

| Component | Fits | Updates per fit |
|---|---:|---:|
| Fine parent: mean and ball-attention, two seeds, three folds | 12 | 7,200 |
| Player-only coarse parent: two seeds, three folds | 6 | 7,200 |
| Control continuation | 12 | 1,200 additional |
| Complete-possession-balanced continuation | 12 | 1,200 additional |
| Masked-teacher continuation | 12 | 1,200 additional |

All three continuations start from the same raw parent and AdamW state. Fine
parents export an EMA over updates 7,001–7,200; continuations export an EMA over
8,201–8,400. The original sampling, losses, learning-rate schedules, augmentation,
ensemble weights, coarse gate, calibration and decoder remain fixed.

The GPU baseline uses the newly rebuilt parent EMAs and coarse models. It is
not the historical v6 checkpoint. Teacher/control training-window transcripts
must match. No assessment scores are calculated until all 54 fits complete.

Evaluation covers:

- Clean, short-gap, long-gap, and target-visibility replay scenarios, with two
  fixed mask batches per assessment set, shared by every method.
- Temporal IoUs 0.1 through 0.9. The official objective averages 0.3, 0.4, 0.5,
  0.6 and 0.7, then weights scenarios 0.50 / 0.20 / 0.15 / 0.15.
- Both fine seeds together and each fine seed separately; both coarse seeds
  remain fixed in all groups.
- Whole-match event precision/recall/F1, missed and spurious possessions,
  duration-binned recall, frame macro F1 and transition F1.
- Paired match differences and descriptive 95% bootstrap intervals, resampling
  six matches rather than frames, masks or seeds.

The existing development-screen thresholds are applied against the fresh GPU
baseline and the matched continuation control. Passing calls for confirmation
and a separate WMX guard, and does not promote a model or alter a submission.
Target data supplies only the previously defined ball-visibility replay pattern.
Challenge possession labels and videos are not needed.

### Assessment-mask correction

The initial GPU protocol selected the earlier campaign search/confirmation
presets. A review during training identified that the historical refinement
study used the later seed-bagging presets. Both assessment designs were frozen
before any new assessment score was calculated. The same 54 fits are evaluated
under both sets; training does not change and no models are selected between
evaluations.

| Assessment | Short seeds | Long seeds | Replay seeds |
|---|---|---|---|
| Historical refinement masks | 73104, 73105 | 74104, 74105 | 5102, 5102 |
| Additional earlier-campaign masks | 73101, 73102 | 74101, 74102 | 5101, 5102 |

Use `historical-masks/report/report.md` for the rerun using the prior study's
mask definitions. The top-level `report/report.md` is the additional robustness
assessment. GPU parents are freshly trained in both cases, so neither is an
exact replay of historical CPU checkpoint values.

## CUDA implementation

`experiments/gpu_runtime.py` adds explicit CUDA session construction and scoped
preprocessing acceleration. The original source-pinned implementation is kept
intact. The existing trainer already supports moving models, windows, targets,
optimizer state and RNG state to CUDA; its historical campaign wrapper selects
CPU explicitly, so the new runner uses the GPU adapter throughout.

The coordinate-noise recurrence is compiled with Numba without fast math.
Feature gap ages and speed medians use equivalent vectorized arithmetic.
Tests check exact values and the NumPy generator state against the original
implementation, including missing observations and discontinuities. Actual CUDA
tests check uninterrupted versus resumed weights, optimizer, sampling counters
and EMA for all three refinement variants. Inference tests require model
parameters to stay on CUDA.

The initial real-data benchmark took about 32.6 seconds for 100 original CUDA
training steps and 4.5 seconds with preprocessing acceleration, after loading
the training halves. This measures one mean-model configuration on a Tesla T4;
it is not a full-study throughput guarantee.

Training uses float32, deterministic algorithms, disabled TF32 and disabled
cuDNN benchmarking. PyTorch does not guarantee identical CPU/GPU trajectories
or reproducibility across different software/hardware environments; the new
protocol records its own environment and source hashes. See the
[PyTorch reproducibility documentation](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

## Run and resume

```bash
uv sync --frozen --extra learn --extra tune --extra dev
# Plotting is optional tooling outside the source-pinned model dependencies.
uv pip install --python .venv/bin/python matplotlib
uv run --no-sync python -u -m experiments.gpu_iou --device cuda --workers 3
uv run --no-sync python -u -m experiments.gpu_iou_historical_masks
```

The default output is `artifacts/gpu-iou-v1/`. Three independent fold workers
share the GPU; each uses one CPU thread and a bounded half cache. Use
`--workers 1` for sequential folds in a separate output directory. Changing
the frozen device, worker count, source, recipe or prepared data requires a new
`--output-dir`.

Every 200 updates, the runner saves model, optimizer, generators, sampling
transcript and counters. Fine-parent EMA tails replay from the saved 7,000-step
state if interrupted before publishing the complete parent transaction.
Refinement checkpoints include their partial EMA. Rerunning the same command
verifies and resumes saved work. A process lock rejects overlapping launches.

The `--stage` options are `plan`, `train`, `evaluate`, and `all` (default).
Evaluation requires the complete authenticated training grid. Status is stored
in `status.json`, with step-level progress in `status-fold0.json` through
`status-fold2.json`. A terminal log is also available at `run.log` for the run
launched from the Codex session.

The historical-mask evaluator can be launched in a second terminal with
`--wait`; it freezes its protocol immediately and waits for the first evaluation
to finish. Its own status, log, input caches and reports are under
`historical-masks/`. It verifies and reuses the same training checkpoints.

Once both assessment input manifests are complete, the optional cache helper
can run alongside evaluation:

```bash
uv run --no-sync python -u -m scripts.reuse_iou_predictions --wait
```

It hard-links predictions only when the input file SHA-256 matches exactly,
and preserves the original checkpoint, fit and device identity. The evaluator
checks that identity and the prediction hash before using a cached result.
Existing files are never replaced. Identical clean and replay inputs can thus
share inference across mask batches; different short/long masks are evaluated
separately. Reuse receipts are saved in `prediction-reuse.json`.

After `training-audit.json` confirms all 54 fits and both historical input
manifests are complete, historical short/long predictions can also be
precomputed while the first assessment runs:

```bash
uv run --no-sync python -u -m scripts.prefetch_iou_predictions --workers 2
```

This helper calls the frozen evaluator's CUDA prediction function and writes
its normal authenticated cache. It performs no scoring or model selection.
Per-fold progress and the helper's source hash are recorded under
`historical-masks/prefetch-fold*.json`.

Read saved-update progress without loading models:

```bash
uv run --no-sync python -m scripts.gpu_iou_status
```

| Output | Contents |
|---|---|
| `protocol.json` | Frozen training design, folds, recipes, data and source hashes |
| `fits/` | Raw states, optimizer/RNG snapshots, completed model weights and fit records |
| `training.json`, `training-audit.json` | Complete grid and actual training-window ancestry checks |
| `assessment/`, `inference/` | Authenticated common inputs and CUDA probabilities |
| `evaluation-protocol.json` | Evaluation source hashes, fixed IoUs and bootstrap design |
| `reference/` | Compact public truth and decoded labels, with hashes |
| `report/report.md` | Readable results after completion |
| `report/result.json` | Full scores, matched comparisons and development screens |
| `report/*.csv` | IoU curves, match metrics and duration recall |
| `report/iou-curves.png`, `report/iou-curves.pdf` | Scientific figures |

Recompute the completed scores from labels alone:

```bash
uv run --no-sync python -m experiments.gpu_iou_evaluation
uv run --no-sync python -m experiments.gpu_iou_historical_masks --score-only
```

This verifies archive/array hashes and compares the independently recomputed
result with the saved result. It requires neither model fitting nor a GPU.

## Frozen follow-up

The follow-up completed on 17 November 2026 under
`artifacts/gpu-iou-followup-v1/`.
Its protocol hash is
`38dfb51664cdc13a0ea7e4ed2f3b0f0d08bc7fb1cc9bcc30418331652fea7712`.
The fixed combined spell-balanced recipe is the confirmation candidate;
the ablation results cannot substitute a different candidate into that test.

| Part | New fits | Training seeds | Assessment |
|---|---:|---|---|
| Fresh fine/coarse confirmation | 42 | 8201, 8202 | Same three folds and six whole matches |
| Sampling-only and loss-only ablations | 24 | Existing 8101, 8102 parents | Same folds, masks, coarse models and decoder |
| WMX guard | 14 | 9201, 9202 | Train on six inner matches; assess WMX separately |

This adds 240,000 optimizer updates. All comparisons use the historical
refinement mask batches. The original combined recipe, learning rates,
1,200-update continuations, and 200-update tail EMA remain fixed. The loss-only
adapter removes the sampling branch's extra RNG draw, so its exact sampled
windows match control. Sampling-only matches the combined recipe's windows.
Both properties and exact CUDA restart were tested. The relevant suite passed
153 tests before assessment. A further CUDA test confirmed exact trajectory
and RNG equality with cache hits versus forced reloads (154 tests in total).

The confirmation repeats the existing six-match gate. The separate WMX guard
requires the candidate's objective and each scenario score to be no worse than
0.01 below both the matched ensemble control and parent baseline. This is a
prespecified follow-up guard, not the older individual-seed bagging gate.
WMX has historical development use and is not an independent test.

The original clean-data error audit is complete: at IoU 0.5, matched events
increase from 1,548 to 1,619 while spurious events increase from 457 to 533.
Operational categories show wrong-team errors +58, splits +27, merges -13,
timing errors +4, and phantom events unchanged. The paired event comparison
recovers 110 true possessions and loses 39; for possessions under two seconds,
it recovers 53 and loses 17. These overlap-based categories are descriptive,
not verified causal explanations. Event-level CSV and a prespecified WOH
timeline are in `error-audit/`.

```bash
uv run --no-sync python -m scripts.iou_followup_status
uv run --no-sync python -m experiments.iou_followup_training --stage train
uv run --no-sync python -m scripts.run_iou_followup_bounded
uv run --no-sync python -m scripts.report_iou_followup
```

The evaluation helper waits for all confirmation and ablation fits, then
overlaps their assessment with WMX training, using at most two inference
workers because the training pool retains memory until WMX finishes.
This scheduling adjustment was recorded before any new scores; all model
and analysis settings remain frozen. Each arm requires a complete,
authenticated fit grid before inference. The input caches are reused by
reference, and identical predictions are reused only when checkpoint and full
input hashes agree. Original artifacts and source-pinned modules are preserved.

WMX's six-match training arrays occupy 1,491,499,841 bytes, exceeding the
initial 1.3 GB cache. Its first fine parent was resumed from the authenticated
update-1,600 snapshot with a 1.6 GB cache using
`python -m scripts.resume_iou_guard`; the idle fold workers were released.
This resource-only amendment preserves the model, optimizer, RNG states,
sampled-window prefix and fixed training budgets. The amendment and the
cache-parity test are recorded in the artifact root.

### Follow-up results

| Assessment | Control | Combined spell balancing | Change | Frozen gate |
|---|---:|---:|---:|---|
| Fresh seeds 8201/8202, six matches | 0.602116 | 0.604912 | +0.002796 | Pass; 5/6 match wins, both batches positive |
| Separate WMX guard, seeds 9201/9202 | 0.537504 | 0.557923 | +0.020419 | Pass; all four scenarios improve |

The fresh-seed six-match gain is smaller than the original +0.007633.
Its descriptive paired-match 95% interval is [+0.000349, +0.005648].
The two individual fine-seed versions decline by 0.003960 and 0.002043;
the confirmation supports the exact two-seed ensemble. Clean-data short-spell
matches rise from 197/728 to 214/728, while spurious events rise from 443 to 540.
WMX is one historically reused match; no confidence interval is estimated.

The factorial ablations use the original parents and therefore have a
different matched control from the fresh-seed confirmation:

| Original-parent condition | Objective | Change versus its control | Match wins |
|---|---:|---:|---:|
| Control | 0.604216 | — | — |
| Sampling only | 0.606709 | +0.002493 | 3/6 |
| Per-possession loss only | 0.607388 | +0.003172 | 3/6 |
| Combined | 0.611849 | +0.007633 | 5/6 |

Both single-component variants fail the four-match breadth requirement.
Their descriptive 95% intervals include zero. The interaction estimate is
+0.001968, with interval [-0.003860, +0.007574], so these data do not establish
synergy. Both components together remain the strongest tested condition, with
more short possessions recovered and more spurious events.

The frozen combined recipe passes confirmation and the WMX guard. This remains
development evidence on historically reused matches, and no delivered model
was changed. New fully labeled matches are the next validation priority.
The complete report and scientific figures are in `report/`; the final
integrity and exact score-reproduction receipt is
`completion-verification.json` in the follow-up artifact root.

## Workflow attribution

The statistical-analysis skill informed the matched-unit and uncertainty
reporting workflow: Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner,
A. M. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for
Research Agents*, [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
The current arXiv record was checked before adding this attribution.

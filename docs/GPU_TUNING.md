# Eight-hour holistic GPU tuning

This campaign follows the verified GPU IoU experiments and their fresh-seed confirmation. It searches for an improvement over the **combined** 8201/8202 ensemble, whose two-batch objective is 0.604912. The original models, data and experiment artifacts are preserved.

The campaign completed successfully on **17 November 2026 at 08:33 UTC**, after approximately **7 hours 50 minutes** on a Tesla T4. It started at **00:43 UTC**, within an eight-hour deadline of **08:43 UTC**, and exited with code 0 without watchdog termination. Preflight passed **97 tests** and an end-to-end CUDA check covering eight tiny-model fits and exact reproduction of 20 saved assessment results. Launch, validation and completion receipts are in the campaign output directory.

The campaign is adaptive development tuning. All seven DFL matches have previous development use. Excluding a match from its model's training prevents direct training leakage, but does not make the repeatedly used development scores an independent test. Fresh training seeds and mask batches are robustness checks, not additional independent matches.

## Completed results

The search completed **23 screening recipes**, **5 full-budget recipes** and **540 decoder trials**. The frozen winner was `trial_0017`, with 9,600 updates per new model. Its selected search objective was 0.628626; the subsequent confirmation used fresh candidate training seeds without changing the selected recipe or decoder.

| Assessment | Incumbent | Candidate | Change | Gate |
|---|---:|---:|---:|---|
| Six-match confirmation, two mask batches | 0.604912 | 0.629368 | +0.024456 | Pass |
| Separate WMX guard, two mask batches | 0.557923 | 0.569078 | +0.011156 | Pass |

These are weighted development aggregates of the official **spellF1** metric, not challenge-match or leaderboard scores. Clean-tracking spellF1 on the six confirmation matches rose from **0.672989 to 0.688007**. The clean WMX score rose from **0.615741 to 0.620046**. The scenario weights are specified below.

Both individual confirmation candidate seeds improved the weighted objective (+0.021813 and +0.019643). All six matches improved in the ensemble aggregate, and all confirmation gate checks passed. Three additional fixed-recipe seeds (9601, 9602 and 9603) also improved over the incumbent by +0.014445, +0.022468 and +0.017992; these diagnostics did not alter selection or the original gate.

At clean IoU 0.5, short possessions recovered increased from **214 to 249 out of 728**, while total spurious predicted possessions increased from **540 to 544**. The descriptive paired whole-match 95% interval for the weighted confirmation gain was [0.018292, 0.030001]. Historical match reuse prevents interpreting it as an unbiased estimate of unseen-match performance.

The final verification reproduced **26 saved-label assessment results** exactly and checked **121 exported checkpoints** plus **92 optimizer states and split transcripts**. Source and upstream artifact hashes were preserved. Detailed local evidence is in `artifacts/gpu-tuning-8h-v1/report.md`, `confirmation.json`, `guard.json`, `stability.json`, `verification.json` and `exit.json`.

The delivered model and `submission/CY-DATA-TEAM.csv` still use **v6**. This campaign has not yet refit and packaged its winning pipeline for challenge inference or generated an updated submission CSV. Data, checkpoints and full run artifacts remain local under the repository's existing ignore rules; this document records the completed results for the public repository.

## Scope and evaluation

The sampled recipe covers mean, ball-attention and interaction pooling; widths 32/48/64; seven to nine temporal dilation layers; 256/512/768-frame training windows; learning rate, weight decay and constant/cosine scheduling; boundary loss and positive weighting; short-possession sampling and per-possession loss; tracking augmentation and ball-outage simulation; 7,200/9,600 full training updates; and raw versus final-200-update EMA exports.

The complete prediction pipeline also tunes the new model's weight against the existing combined fine ensemble, the coarse player-only context weight and gap-length gate, neural/reference mixing, calibration temperatures, asymmetric transition costs, boundary evidence and reference smoothing. The existing coarse models remain fixed. A decoder-only improvement of the incumbent is eligible, so a new neural model must earn its additional complexity.

Every neural proposal is assessed out of fold on all six development matches using the same three train-four/validate-two splits. Both halves are joined with an X gap. The objective is the official possession F1 averaged over temporal IoUs 0.3–0.7, with equal whole-match weights and clean/short/long/replay weights 0.50/0.20/0.15/0.15. The search uses mask batch 1; batch 2 is excluded from this campaign's selection, although it has historical development use.

The incumbent was reproduced exactly before launch: batch-1 objective **0.6044802814086043**, **214/728** short possessions matched, **540** spurious predictions at clean IoU 0.5.

## Time allocation

One persistent eight-hour wall-clock budget includes training, inference, scoring and report generation. Restarting does not reset the deadline.

| Stage | Deadline after launch | Work |
|---|---:|---|
| Screening | 2 h 14 min | Seeded TPE proposals; 1,800 updates per fold; existing recipes included as anchors |
| Promotion | 4 h 10 min | Architecture-diverse shortlist, 3,600-update checks and full training; additional proposals while time remains |
| Pipeline tuning | 4 h 58 min | Joint blending/decoder search on the best three completed full recipes and the incumbent |
| Confirmation | 7 h 12 min | Freeze one winner; train candidate seeds 9401/9402; evaluate both mask batches |
| WMX guard | 7 h 50 min | Train the selected candidate on the six inner matches with seeds 9501/9502; assess WMX |
| Extra seed checks | Remaining time to 7 h 50 min | When main assessments finish early, repeat the fixed candidate with seeds starting at 9601; diagnostics only |
| Verification/report | 8 h | Verify checkpoints, split transcripts and saved-label scores; write the report and figures |

The number of completed proposals depends on measured throughput. An incomplete training configuration cannot win. If time is exhausted, completed results and resumable states remain available and the report identifies the unfinished stage. The supervisor requests termination at the deadline and allows a brief checkpoint grace period.

Confirmation keeps the existing incumbent components fixed and repeats the selected **new candidate** with fresh seeds. It tests the addition and selected decoder conditional on that incumbent, rather than retraining every incumbent component. Both individual candidate seeds, their ensemble and a decoder-only decomposition are reported. The frozen confirmation gate requires ensemble gain greater than 0.002, positive gains for both seeds and both batches, improvement on at least four of six matches, positive missing-ball gain, and no scenario loss worse than 0.01. WMX requires no objective/scenario loss worse than 0.01. Short-event recall and spurious counts remain explicit diagnostics. There is no automatic production promotion.

## Operation and storage

If a decoder-only pipeline wins, its models stay fixed and candidate-seed checks are inapplicable; the match, mask and WMX checks still apply. Additional seed diagnostics never change the winner or the original confirmation gate.

Launch from the repository root:

```bash
uv run --no-sync python -m scripts.launch_gpu_tuning
uv run --no-sync python -m scripts.gpu_tuning_status
```

The launcher creates a source snapshot and starts a detached Linux supervisor. Training and inference require CUDA; they fail instead of silently falling back to CPU. Preprocessing, decoder search and official scoring use CPU code. The existing tested float32 deterministic CUDA adapter is retained.

Files are stored under `artifacts/gpu-tuning-8h-v1/`. The frozen source snapshot is under `artifacts/runtime/gpu-tuning-8h-v1/`.

- `status.json`, `clock.json`, `supervisor.json`, `stdout.log`: live stage, deadline and process details.
- `protocol.json`, `proposals/`, `screen/`, `middle/`, `full/`, `polish/`: complete recipes and scored search history.
- `fits/`: model checkpoints, optimizer/RNG state, training-window transcript hashes and split provenance.
- `selection.json`: immutable winner, written before confirmation or WMX scoring.
- `confirmation.json`, `guard.json`: final comparisons and diagnostic counts.
- `confirmation/`, `guard/`: saved labels and exact scores for each seed, ensemble, incumbent and decoder-only comparison.
- `verification.json`, `report.md`, `confirmation-iou.png`, `confirmation-iou.pdf`: final audit and report.

The disk budget retains one committed optimizer state per completed fit and two while updates are in progress. Only this new campaign's obsolete rolling states are removed. Predictions stay in memory; final decoded labels are saved. Data and input caches are reused directly. A 650 MB free-space reserve stops further work if reached.

After an interruption, inspect `status.json`, `error.json` and `exit.json`, then resume the **same frozen snapshot and original deadline** with:

```bash
uv run --no-sync python -m scripts.launch_gpu_tuning --resume
```

The functional preflight uses tiny 200-update models in `artifacts/gpu-tuning-integration-v1/`. Those scores are software checks and never enter the real search.

## Design attribution

The statistical-analysis and scientific-critical-thinking skills informed the whole-match comparisons, historical-reuse caveat and separation of selection from confirmation. Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065). The current arXiv record was checked on 17 November 2026.

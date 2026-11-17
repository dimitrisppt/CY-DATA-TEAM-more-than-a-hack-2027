<div class="eyebrow">CY-DATA-TEAM · More Than a Hack 2027</div>

# Football possession from incomplete tracking

<div class="subtitle">Technical report · 17 November 2026</div>

**Team members**<br>
Dimitris Papatheodoulou · Evripides Alevras · Andreas Petridis

## 1. Overview and objective

We built an offline system that converts automatic football tracking into a
continuous sequence of home possession, away possession and dead play. The
central challenge is temporal: proximity to the ball is useful evidence, but a
nearby player is not necessarily in control, and a missing ball observation
does not mean play has stopped. Our approach combines learned temporal
context, simple geometric evidence and a sequence decoder that discourages
unsupported changes of state.

The **v7 submission** uses eight networks: four possession-balanced fine
models, two additional attention models selected by GPU tuning, and two
player-only models at 1 Hz. Fine models resolve local play; coarse models
supply longer context during ball gaps. We selected the complete pipeline
using possession-level scoring and refit it on all seven public DFL matches.
Both halves of each match stay together in development validation folds.

> **Delivered model.** `submission/CY-DATA-TEAM.csv` is v7 and includes the
> confirmed possession-balancing and eight-hour GPU-tuning improvements.
> The previous v6 CSV is preserved under `submission/archive/v6/`.
> Reported development scores come from match-excluded validation fits;
> they are not measurements of the final CSV on the challenge match.

### What the system must output

The challenge requires **199,319 rows** at 25 frames per second, indexed from
0 to 199318. Each row contains one label: `H` for home possession while live,
`A` for away possession while live, `D` for dead play, or `X` for no answer.
Each half is processed independently. We write `X` outside the published
scoring spans, including halftime, and produce `H/A/D` within them. [1]

### Why we optimize possessions rather than frame accuracy

The ranking metric, **spellF1**, matches predicted and true possessions
one-to-one by temporal intersection-over-union (IoU). A possession is a maximal
run of `H` or `A`; a `D` interval separates possessions. F1 is averaged over
IoU thresholds **0.3, 0.4, 0.5, 0.6 and 0.7**. Per-frame macroF1 is a secondary
metric and the first tie-break. [1]

This changes the modeling objective. A predictor can label the middle of a
long possession correctly yet create several false possessions around a
transition. Strong frame accuracy can therefore coexist with weak spellF1.
We explicitly examine interval boundaries, short possessions and spurious
events. During scoring, predicted `X` values are filled within each half;
they are not a way to avoid temporal errors. Our missing-ball strategy instead
uses the remaining tracking evidence.

<!-- pagebreak -->

## 2. Data, labels and possession semantics

### Sources and their roles

| Source | Scope | Use in this project |
|---|---|---|
| Sportec Solutions / DFL dataset [2] | Seven Bundesliga matches; fourteen halves; 1,002,644 frames | Supervised training, model selection and development evaluation |
| SkillCorner Open Data [3] | Two matches; 47,121 eligible proxy frames out of 118,494 | Earlier conditional home/away diagnostic; no submission training or selection |
| Challenge tactical tracking | Automatic tracking text and metadata XML | Input to challenge inference; unlabeled visibility patterns also inform outage assessment |

The DFL release is provided by Sportec Solutions with DFL authorization under
CC BY 4.0. We use the supplied position-data fields `BallStatus` and
`BallPossession`: inactive play becomes `D`; known ownership during active
play becomes `H` or `A`; unavailable supervision is masked. The team created
no manual labels. The source identifies the data provider, but the records
reviewed do not establish individual annotators or a field-specific
adjudication workflow for these two targets. We therefore attribute the
labels to the provider without inventing annotator details. [2]

SkillCorner agreement labels cover a conditional subset and have no corrected
dead-play state. They cannot measure full three-state possession performance.
We retained that diagnostic as a limited transfer check, rather than treating
it as a substitute for fully labeled validation matches.

### Assets used, and why

Challenge inference reads `BAR-ATH_tactical_atd.txt` and
`BAR-ATH_tactical_atd.xml`. Tracking already supplies geometry, so we focused
our effort on ownership, timing and robustness instead of rebuilding object
detection. The inference pipeline does not use tactical or broadcast video,
broadcast tracking, Smart Tagging, homography, or challenge possession labels.
Broadcast alignment and additional visual or tagging evidence remain possible
extensions; they are not implemented components of v7.

The five-minute public challenge labels were not used for training or model
selection. During development, we replayed automatically measured challenge
ball-visibility patterns on labeled DFL halves. This tests sensitivity to the
target's missing-input pattern without using target possession labels.
Participant assets and visibility arrays are excluded from the public
validation reference. Inference runs locally and calls no external model API.

### How possession rules enter the algorithm

The network predicts live/dead status separately from ownership conditional
on live play. Dead-ball ownership is not a target. Missing ball observations
are represented by masks and gap durations, not converted into `D` labels.

For loose balls and challenges, temporal context and switch penalties can
preserve the previous team when evidence for a handover is weak. This is a
learned estimate, not an explicit guarantee of the challenge's controlled
possession rule. We do not hard-code a minimum number of touches or passes.
At stoppages and restarts, learned live/dead and boundary evidence drive
transitions to and from `D`. Brief deflections, aerial duels and ambiguous
restarts remain failure cases because source-provider conventions may differ
from the challenge definition.

<!-- pagebreak -->

## 3. The v7 prediction pipeline

### Features and preprocessing

Coordinates are pitch-normalized. Eight player features encode position,
valid motion, motion validity, ball-relative position and visibility. Nine
global features describe ball position/motion, observation masks, gap ages and
median player speed. Invalid tokens are masked. Motion requires adjacent frames
and plausible speed. Inputs contain no labels or player-identity embeddings.

### Fine model structure

Shared encoders process each team's unordered player set. Mean pooling
summarizes team structure; ball-conditioned attention emphasizes controller
evidence and falls back to mean pooling when the ball is unseen. Dilated
convolutions add temporal context without requiring stable player identities.
The width-32 architectures have eight blocks and a **511-frame (20.44-second)**
receptive field. Added width-64 attention uses seven blocks and **255 frames
(10.2 seconds)**. Neural heads predict live play, conditional ownership and
boundaries, sharing parameters across exchanged team orderings.

Two seeds per architecture produce six fine models. Live/ownership outputs
are mixed in logit space; boundary outputs are averaged linearly. The original
pair uses mean-pooling weight **0.619862**. The added attention pair receives
weight **0.203128** against that ensemble. Seeds reduce reliance on one fit,
without creating independent validation matches.

### Missing-ball fusion

Two width-32 player-only models operate at 1 Hz with a **255-sample context**,
about 4.25 minutes. Ball inputs are always hidden during their training and
inference. Their live and ownership logits are interpolated back to 25 Hz and
blended with the fine branch only when the ball is missing:

**Coarse weight = 0.318357 × min(d / 3.218862, 1)**, where `d` is the distance in seconds
to the nearest observed ball within the same half. The weight is zero on
observed-ball frames. Fine boundary evidence is preserved.

Player movement supplies context during gaps. Both branches use past and
future frames; the method is offline.

### From probabilities to a coherent sequence

A geometric reference uses team-ball proximity for ownership, and player
activity, valid ball motion and pitch boundaries for live/dead evidence.
Neural mixing weight is **0.447310**; live/ownership temperatures are
**3.254195/0.517258**; boundary scale is **1.580429**. Reference smoothing and
transition costs are frozen in the recipe.

If `p` is live probability and `q` is home ownership given live play, the state
probabilities are `P(H) = p × q`, `P(A) = p × (1 − q)` and `P(D) = 1 − p`.
A three-state Viterbi decoder chooses the best sequence using log-emissions,
boundary evidence and switch costs. This penalizes unsupported fragmentation
while permitting genuine transitions. It runs separately for each half,
without smoothing across halftime.

<!-- pagebreak -->

## 4. Training and validation design

### Training choices

All eight release models are fitted on the fourteen public DFL halves.
The four original fine models receive **7,200 parent updates plus 1,200
possession-balanced continuation updates**. The two coarse models receive
**7,200 updates**; the two tuned attention models receive **9,600 updates**.
Training uses AdamW, masked live/owner binary cross-entropy and boundary loss.
Ownership requires known live play; boundary targets require valid adjacent
states. Central windows are scored with surrounding context retained.

Ordinary frame loss overweights long possessions. The continuation adds
per-possession loss weight **0.25** and short-possession sampling probability
**0.25**, with learning rate reduced to 20% of the parent rate. Its fine exports
use EMA with decay **0.99** over updates **8,201–8,400**. Tuned attention uses
per-possession weight **0.25**, sampling probability **0.50**, a 768-frame center,
constant learning rate **0.000862334**, and raw update-9,600 weights.

Training includes synthetic outages, team exchange and selected coordinate
noise or player dropout. Final seeds were fixed before target inference: **20261117/20261118** for original fine/coarse
models and **20261119/20261120** for tuned attention. Original-branch seeds use
the recorded deterministic mapping. Checkpoints and sampled-window transcripts
authenticate the public training split and update counts.

### Validation and results

The sampling unit is a **whole match**. Six development matches form three
train-four/validate-two folds; both halves and all windows stay together. A
separate WMX check fits on the six other matches and evaluates WMX. Final
all-seven delivery fits are separate from the validation predictions.

Each assessment includes clean tracking, short synthetic ball gaps, long gaps
and replayed visibility patterns. We use the official spellF1 scorer and
equal match weights. The development objective is:

**J = 0.50 × clean + 0.20 × short + 0.15 × long + 0.15 × replay.**

These scenario weights are our development choice, not the competition's
leaderboard aggregation. The missing-ball score renormalizes the last three
terms. Two fixed outage batches and paired seeds probe robustness. All seven
matches have historical development use: exclusions prevent direct training
leakage, but cannot remove selection bias from repeated experimentation.

| Historical matched v4–v6 comparison | Weighted J | Missing-ball score |
|---|---:|---:|
| v4: four-network ensemble | 0.580751 | 0.512152 |
| v5: fine-model tail averaging | 0.590127 | 0.525637 |
| **v6: added player-only context** | **0.594200** | **0.533782** |

Both changes passed their gates. Coarse context improved missing-ball scores
with unchanged clean-scenario arrays. It adds capacity as well as temporal
range; this comparison does not isolate the cause of improvement.

### Experiments that informed the design

Time-shift augmentation, interaction pooling and decoder-only refinement
failed relevant repeat or advancement checks. Contextual timing adjustment
gained only 0.000275, below its threshold. These results informed the design
but were not promoted into the final pipeline. [4]

<!-- pagebreak -->

## 5. GPU evidence behind v7

**V7 incorporates the confirmed combined refinement and frozen tuning winner.**
Each comparison uses its own matched controls, masks and seeds. The v4–v6
scores and GPU-study scores are separate comparisons, not one leaderboard.

### Short-possession refinement

Ordinary frame losses give long possessions more weight. We tested extra
training with increased short-possession sampling and a per-possession loss,
alongside a matched continuation control and masked-teacher supervision.
A fresh CUDA reconstruction completed **54 fits** and swept temporal IoUs
from 0.1 to 0.9; the official objective still uses 0.3–0.7.

Combined possession balancing scored **0.611849**, versus **0.604216** for its
control. Masked-teacher consistency did not pass. An **80-fit follow-up** then
tested fresh seeds, separate sampling/loss ablations and WMX. The combined
two-seed recipe scored **0.604912 versus 0.602116** in confirmation and
**0.557923 versus 0.537504** on WMX. Sampling-only and loss-only variants each
won three of six matches and failed the four-match breadth gate.

Both individual fine-seed comparisons declined in that confirmation, so the
finding supports the exact ensemble rather than a universal seed-independent
gain. More short possessions were recovered, but spurious predictions also
increased. This motivated joint tuning of training and decoding. [4]

### Holistic tuning: 7 hours 50 minutes on a Tesla T4

The tuning campaign jointly searched architecture, training windows, optimization,
augmentation, outages, losses, sampling, weight averaging, ensemble blending,
coarse gating, calibration and transition costs. Incumbent and coarse models
remained fixed during search and confirmation. CUDA used deterministic float32 execution
and resumable optimizer/RNG states to make interrupted runs auditable.

Seeded Optuna TPE and staged training completed **23 screening recipes**,
**5 full-budget recipes** and **540 decoder trials** within the eight-hour
allocation. The frozen winner, `trial_0017`, adds a width-64 ball-attention
model with seven temporal blocks, a 768-frame training center and 9,600
updates. It uses raw weights, a constant learning rate of approximately
0.000862, and per-possession loss weight 0.25. The pipeline was frozen before
fresh-candidate-seed confirmation and WMX assessment.

| Assessment, two mask batches | Incumbent J | Candidate J | Change |
|---|---:|---:|---:|
| Six-match fresh-candidate-seed confirmation | 0.604912 | **0.629368** | +0.024456 |
| Separate WMX guard | 0.557923 | **0.569078** | +0.011156 |

Both confirmation seeds improved, all six match aggregates improved, and the
confirmation and WMX gates passed. Three additional fixed-recipe seed checks
also improved without changing selection. Clean six-match spellF1 rose from
0.672989 to **0.688007**. At clean IoU 0.5, short possessions recovered rose
from **214 to 249 out of 728**, while total spurious possessions rose from
**540 to 544**.

The descriptive paired-match 95% interval for the confirmation gain was
[0.018292, 0.030001]. Reused matches and adaptive search prevent interpreting
it as an unbiased estimate on unseen matches. Final integrity checks reproduced
26 saved-label assessments exactly and checked 121 exported checkpoints plus
92 optimizer states and split transcripts. These checks support computational
integrity, not external validity. [4]

<!-- pagebreak -->

## 6. Limitations and reproduction

The largest uncertainty is transfer: seven repeatedly studied matches are
insufficient to establish generalization to the challenge. Label conventions
may differ across providers; long ball outages remove direct control evidence;
one-hertz context discards brief motion; and fragmented or switched tracking
slots can corrupt velocity cues. Our offline method also uses future context.
The next priorities are new fully labeled matches, targeted analysis of false
short possessions and controlled tests of added model capacity. Video or
tagging evidence would require a separate implementation and validation.

### Reproduction

The repository is
[dimitrisppt/CY-DATA-TEAM-more-than-a-hack-2027](https://github.com/dimitrisppt/CY-DATA-TEAM-more-than-a-hack-2027).
Use Python **3.12.11**, the committed `uv.lock`, and the eight-model `models/v7/`
bundle. GPU inference is the default; CPU inference is also available.

```sh
uv sync --frozen --python 3.12.11 --extra learn --extra tune --extra dev
uv run --no-sync python -m scripts.reproduce_v7 --verify-only
uv run --no-sync python -m scripts.reproduce_v7 \
  --data-dir /path/to/participant-data \
  --output outputs/CY-DATA-TEAM.csv
uv run --no-sync python evaluation/validate_submission.py \
  outputs/CY-DATA-TEAM.csv
```

The input directory contains the tactical text/XML pair from Section 2. The
runner verifies bundled files and source hashes, reads the two automatic
inputs, processes halves separately, validates the output and checks the
release CSV checksum. Tesla T4 GPU and Linux CPU runs regenerated identical
bytes in the pinned environment. Format validation alone does not establish
accuracy. The delivered **199,319-row v7 CSV** has SHA256:

```text
d187fcf9f3840cd1391f2089dc678f07de4a7760ae09b016ce8da15c61236170
```

> **Release verification.** The v7 bundle contains its eight inference weights,
> frozen recipe and training receipts, without depending on v6's incomplete
> historical proofs. The previous CSV and PDF are archived. Exact regeneration
> and hardware checks are recorded in `submission/verification-v7.json`.
> A public repository URL remains required by the organizers; local artifact
> verification does not establish public availability or submit the entry.

Run `uv run --no-sync python -m scripts.evaluate_v7` to recalculate the two
GPU confirmation tables from saved public labels and predictions, without
training or participant data. `scripts.refit_v7` performs full public-data
fitting from the frozen recipe; it does not repeat hyperparameter search.
See [4] for runtime details, historical v6 checks and reproduction limits.
New hardware can change floating-point decisions; the CSV hash detects this.
Participant match assets are excluded from the release.

### Sources and implementation record

1. Metrica Sports. *More Than a Hack 2027*: [challenge](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/CHALLENGE.md),
   [scoring specification](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/EVALUATION.md)
   and [submission requirements](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md).
2. Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). *An integrated dataset
   of spatiotemporal and event data in elite soccer*. Scientific Data 12, 195.
   [Paper](https://doi.org/10.1038/s41597-025-04505-y);
   [dataset v1, CC BY 4.0](https://doi.org/10.6084/m9.figshare.28196177.v1).
3. SkillCorner. [Open Data repository](https://github.com/SkillCorner/opendata).
   Used only for the earlier proxy diagnostic described in Section 2.
4. CY-DATA-TEAM. [Experiment history](EXPERIMENTS.md),
   [GPU IoU study](GPU_IOU_EXPERIMENT.md), [GPU tuning](GPU_TUNING.md),
   [reproduction details](REPRODUCIBILITY.md) and [data policy](DATA_POLICY.md).
   The v7 recipe, training receipts and model/source hashes identify the delivered implementation.
5. Kassis, T., Agarwal, V., He, Y., Patel, D. & Brueckner, A. M. (2026).
   *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*.
   [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
   Scientific-critical-thinking guidance informed evidence review and reporting.

# Data use and release contents

## Participant match

The final inference pipeline reads only the tactical automatic tracking text
and its XML metadata. Each input is verified against a manifest before use.
All parsing, feature extraction and inference run locally. No network service
is called by the reproduction command.

The participant videos, broadcast tracking, Smart Tagging and homography are
not model inputs. The challenge possession truth and public five-minute
labels are not used for model training or selection. Outputs come from the
algorithm, with no manual possession labels, tracking corrections or
frame-specific output corrections. Published scoring intervals determine
which frames are excluded, including halftime.

During development, automatically measured ball-visibility patterns from the
participant tracking were replayed over labeled public matches. This measures
robustness to an unlabeled input distribution. It does not create or use
participant possession labels. Target visibility arrays are not part of the
public validation reference.

## Public datasets

Supervised learning uses seven other matches from the Sportec Solutions/DFL
release. Labels are the supplied provider fields, mapped to the project's
three supervised states. Unknown states are masked. An earlier SkillCorner
diagnostic used conditional home/away agreement, with no dead-play labels.
It was not a source of v6 or v7 training labels or selection scores.

See [the technical report](CY-DATA-TEAM.md) for quantities, label semantics and
limitations, and [third-party notices](../THIRD_PARTY_NOTICES.md) for
attribution and source licenses.

## Public release boundary

The release contains source, pinned dependencies, trained models, public-data
validation proofs, a compact public label reference, and the algorithmic
submission CSV. Participant tracking, metadata, videos, tagging, homography,
private labels and local experiment directories are excluded.

`data/`, `artifacts/` and `outputs/` are ignored by Git. The release builder
uses an explicit allowlist and verifies file hashes. Its output inventory is
recorded in `package-manifest.json`. Do not add participant assets to that
allowlist or commit derived exports that reconstruct them.

The [challenge rules](../RULES.md), [data documentation](../DATA.md), and terms
accepted at enrollment govern participant asset use and permitted processing
destinations. Source-code licensing does not grant rights to those assets.
This document describes the implemented pipeline and release contents. It
does not replace the organizers' terms or certify prior data handling.

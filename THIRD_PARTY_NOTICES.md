# Sources and attribution

## Challenge repository

The original challenge specification, evaluation code and baseline materials
come from [Metrica Sports, More Than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027).
The upstream revision used here is `f8f2629ff1e0985f81d7e59d8413678514ee1530`.
Its copyright notice and MIT license are retained in [LICENSE](LICENSE).
Competition tracking, video, tagging and homography have separate terms and
are not redistributed with this project.

## Public training data and validation reference

The trained models use the Sportec Solutions/DFL dataset described by:

Manuel Bassek, Robert Rein, Hendrik Weber and Daniel Memmert (2025).
*An integrated dataset of spatiotemporal and event data in elite soccer.*
Scientific Data 12, 195.
[Paper and data description](https://doi.org/10.1038/s41597-025-04505-y).
[Dataset, version 1](https://doi.org/10.6084/m9.figshare.28196177.v1).

The dataset is released under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
The source contains seven matches. Sportec Solutions provides the match data,
with DFL authorization described in the publication. This project uses the
provider's position-data state fields as supervision. It does not claim to
have created those source labels.

The label archives in `benchmarks/v6/public-reference/`,
`benchmarks/timing-calibration-v1/`, and `benchmarks/partial-refinement-v1/`
contain transformed provider labels and predictions produced by this project.
The changes are: mapping source
states to integer `H/A/D/X` codes, splitting timelines by half, and storing
decoded model outputs for the declared validation scenarios. It contains no
raw coordinates or participant-match assets. The transformed source labels
retain their CC BY 4.0 attribution. Prediction files and trained checkpoints
are outputs of this project's code. No endorsement by the dataset authors,
Sportec Solutions or DFL is implied.

## Earlier external diagnostics

[SkillCorner Open Data](https://github.com/SkillCorner/opendata) was used for
an earlier conditional home/away agreement diagnostic. That repository carries
an [MIT license](https://github.com/SkillCorner/opendata/blob/master/LICENSE).
Those matches did not train or select v6. SkillCorner raw data and labels are
not included in the release.

## Software

The project uses NumPy, pandas, PyTorch, Optuna, scikit-learn and Numba.
Pytest and Ruff support development. Versions and distribution hashes are
pinned in [uv.lock](uv.lock). These dependencies retain their own licenses.
The release archive contains project source and artifacts, not copies of the
installed third-party Python distributions. Install them with the documented
`uv sync --frozen` command.

The repository includes the original Metrica scorer. The possession models,
augmentation policies, validation protocols and reproduction scripts are
implemented here. Literature suggestions that were not implemented are
identified as such in [the experiment history](docs/EXPERIMENTS.md).

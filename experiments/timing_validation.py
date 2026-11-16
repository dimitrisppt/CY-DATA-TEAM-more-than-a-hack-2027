"""Match-grouped timing study plans and fail-closed prediction ancestry checks.

These helpers read metadata only. Hashes bind declared provenance to artifacts,
but cannot establish that a producer truthfully declared every fitted ancestor.
Producers must enumerate fine, coarse, feature normalization and all learned
reference components. Ordinary OOF caches are not safe for nested calibration.
Historical recipe selection is reported separately from direct fit leakage.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PREDICTION_SCHEMA = "timing_prediction_provenance_v1"
PROTOCOL_SCHEMA = "timing_validation_protocol_v1"
COMPONENT_KINDS = frozenset({"fine_model", "coarse_model", "normalization",
                             "learned_reference", "calibrator", "feature_transform",
                             "ensemble", "deterministic"})
HISTORICAL_REUSE_CAVEAT = (
    "Historical hyperparameter and recipe selection reused development matches. "
    "Excluding direct training ancestry does not undo that reuse. Results are "
    "development evidence, including any historically reused guard match."
)


@dataclass(frozen=True)
class ComponentProvenance:
    name: str
    kind: str
    train_matches: list[str]
    selection_matches: list[str]
    fit_sha256: str
    model_sha256: str
    ancestors: list[ComponentProvenance] = field(default_factory=list)


@dataclass(frozen=True)
class ConfigLineage:
    name: str
    sha256: str
    training_matches: list[str]
    selection_matches: list[str]
    historical_reuse: bool = False


@dataclass(frozen=True)
class BasePredictionProvenance:
    match: str
    half: int
    scenario: str
    split_id: str
    feature_sha256: str
    components: list[ComponentProvenance]
    config_lineage: list[ConfigLineage]
    schema: str = PREDICTION_SCHEMA
    ancestry_complete: bool = True


def canonical_sha256(value: Any) -> str:
    """Hash finite, canonical JSON. List order remains significant."""
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _canonical_bytes(value):
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _object(value, keys, context):
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ValueError(f"{context}: missing or unrecognized provenance fields")


def _identifier(value, context):
    if (not isinstance(value, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value)
            or re.search(r"_half\d", value, re.IGNORECASE)):
        raise ValueError(f"{context}: invalid identifier {value!r}")
    return value


def _matches(values, context, *, nonempty=False):
    if not isinstance(values, (list, tuple, set, frozenset)):
        raise ValueError(f"{context}: match IDs must be a collection")  # noqa: TRY004
    result = [_identifier(value, context) for value in values]
    if len(result) != len(set(result)) or (nonempty and not result):
        raise ValueError(f"{context}: duplicate or missing match IDs")
    return set(result)


def _hash(value, context):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{context}: missing or invalid SHA-256")


def validate_prediction_provenance(record, *, outer_assessment_matches,
                                   allowed_matches=None, expected_components=None):
    """Validate every training and selection ancestor, returning an audit.

    The sample itself and all outer assessment matches are forbidden in fitted
    components, current config training and current selection. Historical config
    selection may overlap and is explicitly reported. Historical *training* is
    never exempt. Unknown fields, kinds and incomplete fit records fail closed.
    ``expected_components`` can enforce an independently declared model inventory.
    """
    if isinstance(record, BasePredictionProvenance):
        record = asdict(record)
    _object(record, BasePredictionProvenance.__dataclass_fields__, "prediction")
    if record["schema"] != PREDICTION_SCHEMA or record["ancestry_complete"] is not True:
        raise ValueError("prediction: unknown schema or incomplete ancestry")
    match = _identifier(record["match"], "match")
    if type(record["half"]) is not int or record["half"] not in (1, 2):
        raise ValueError("prediction: half must be 1 or 2")
    for key in ("scenario", "split_id"):
        _identifier(record[key], key)
    _hash(record["feature_sha256"], "features")
    outer = _matches(outer_assessment_matches, "outer assessment")
    forbidden = outer | {match}
    allowed = None if allowed_matches is None else _matches(allowed_matches, "allowed matches")
    if allowed is not None and not forbidden <= allowed:
        raise ValueError("prediction: assessment/sample IDs outside declared match universe")
    components = record["components"]
    if not isinstance(components, list) or not components:
        raise ValueError("prediction: missing fitted component provenance")
    top_names = [component.get("name") if isinstance(component, Mapping) else None
                 for component in components]
    if len(set(top_names)) != len(top_names):
        raise ValueError("prediction: duplicate component names")
    if expected_components is not None and set(top_names) != set(expected_components):
        raise ValueError("prediction: required component inventory differs")
    visited = set()
    training, selection = set(), set()

    def check_matches(values, context):
        ids = _matches(values, context)
        if allowed is not None and not ids <= allowed:
            raise ValueError(f"{context}: unknown match IDs")
        return ids

    def walk(component, stack):
        _object(component, ComponentProvenance.__dataclass_fields__, "component")
        identity = id(component)
        if identity in stack:
            raise ValueError("component: cyclic ancestry")
        _identifier(component["name"], "component name")
        if component["kind"] not in COMPONENT_KINDS:
            raise ValueError("component: unrecognized ancestry kind")
        for key in ("fit_sha256", "model_sha256"):
            _hash(component[key], key)
        fit_ids = check_matches(component["train_matches"], "component training")
        selected = check_matches(component["selection_matches"], "component selection")
        if not fit_ids and component["kind"] != "deterministic":
            raise ValueError("component: learned component has absent fit provenance")
        if component["kind"] == "deterministic" and (fit_ids or selected):
            raise ValueError("component: deterministic ancestry declares learned matches")
        if forbidden & (fit_ids | selected):
            raise ValueError("direct or indirect leakage in component ancestry")
        training.update(fit_ids)
        selection.update(selected)
        ancestors = component["ancestors"]
        if not isinstance(ancestors, list):
            raise ValueError("component: missing explicit ancestry list")  # noqa: TRY004
        if identity not in visited:
            for ancestor in ancestors:
                walk(ancestor, stack | {identity})
            visited.add(identity)

    for component in components:
        walk(component, set())
    configs = record["config_lineage"]
    if not isinstance(configs, list) or not configs:
        raise ValueError("prediction: missing configuration lineage")
    historical = set()
    names = set()
    for config in configs:
        _object(config, ConfigLineage.__dataclass_fields__, "configuration")
        name = _identifier(config["name"], "configuration name")
        if name in names:
            raise ValueError("configuration: duplicate names")
        names.add(name)
        _hash(config["sha256"], "configuration")
        trained = check_matches(config["training_matches"], "configuration training")
        selected = check_matches(config["selection_matches"], "configuration selection")
        if type(config["historical_reuse"]) is not bool:
            raise ValueError("configuration: historical_reuse must be boolean")
        if trained & forbidden or (not config["historical_reuse"] and selected & forbidden):
            raise ValueError("direct or indirect leakage in configuration lineage")
        training.update(trained)
        if config["historical_reuse"]:
            historical.update(selected)
        else:
            selection.update(selected)
    return {"match": match, "training_matches": sorted(training),
            "selection_matches": sorted(selection),
            "historical_selection_matches": sorted(historical),
            "historical_reuse_caveat": HISTORICAL_REUSE_CAVEAT if historical else None,
            "provenance_sha256": canonical_sha256(record)}


def validate_prediction_records(records, *, outer_assessment_matches,
                                expected_matches=None, allowed_matches=None,
                                expected_components=None):
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("prediction records must be nonempty")
    audits = [validate_prediction_provenance(
        row, outer_assessment_matches=outer_assessment_matches,
        allowed_matches=allowed_matches, expected_components=expected_components) for row in records]
    if expected_matches is not None and {row["match"] for row in audits} != _matches(
            expected_matches, "expected matches"):
        raise ValueError("prediction records do not cover expected matches")
    keys = [(row.match, row.half, row.scenario) if isinstance(row, BasePredictionProvenance)
            else (row["match"], row["half"], row["scenario"]) for row in records]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate prediction sample")
    return audits


def validate_group_assignments(records):
    """Reject splitting a match's scenarios or halves across dataset roles."""
    groups = {}
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("group assignments must be nonempty")
    for row in records:
        _object(row, ("match", "half", "scenario", "split"), "assignment")
        match = _identifier(row["match"], "match")
        if type(row["half"]) is not int or row["half"] not in (1, 2):
            raise ValueError("assignment: invalid half")
        _identifier(row["scenario"], "scenario")
        split = _identifier(row["split"], "split")
        if match in groups and groups[match] != split:
            raise ValueError("same match scenarios or halves cross dataset splits")
        groups[match] = split
    return groups


def _requests(training, assessment, forbidden, prefix):
    """Cross-fit calibration inputs, then fit base models for assessment."""
    training, assessment, forbidden = set(training), set(assessment), set(forbidden)
    if training & assessment or (training | assessment) & forbidden:
        raise ValueError("split groups overlap")
    result = []
    for match in sorted(training):
        result.append({"id": f"{prefix}-calibration-{match}", "role": "calibration",
                       "predict_matches": [match], "train_matches": sorted(training - {match}),
                       "forbidden_matches": sorted(forbidden | assessment | {match})})
    result.append({"id": f"{prefix}-assessment", "role": "assessment",
                   "predict_matches": sorted(assessment), "train_matches": sorted(training),
                   "forbidden_matches": sorted(forbidden | assessment)})
    return result


def build_nested_plan(matches, *, search=False, outer_assessment_folds=None,
                      historical_selection_matches=()):
    """Predeclare 3 outer folds over 6 whole matches.

    Fixed mode evaluates a configuration declared before assessment, without
    inner HPO. Search mode adds leave-one-match-out selection inside each set
    of four. Each inner calibration sample then uses a base fit on two matches.
    Every learned ancestor must follow the corresponding request exclusion.
    """
    ids = sorted(_matches(matches, "development matches", nonempty=True))
    if len(ids) != 6 or type(search) is not bool:
        raise ValueError("nested timing protocol requires six matches and boolean search")
    groups = ([ids[index:index + 2] for index in range(0, 6, 2)]
              if outer_assessment_folds is None else outer_assessment_folds)
    if not isinstance(groups, (list, tuple)) or len(groups) != 3:
        raise ValueError("exactly three assessment folds are required")
    groups = [sorted(_matches(group, "assessment fold")) for group in groups]
    coverage = sorted(match for group in groups for match in group)
    if any(len(group) != 2 for group in groups) or coverage != ids:
        raise ValueError("assessment folds must cover six matches exactly once")
    historical = sorted(_matches(historical_selection_matches, "historical selection"))
    folds = []
    for index, assessment in enumerate(groups):
        training = sorted(set(ids) - set(assessment))
        prefix = f"outer-{index}"
        fold = {"id": prefix, "training_matches": training, "assessment_matches": assessment,
                "selection_matches": training if search else [],
                "prediction_requests": _requests(training, assessment, [], prefix),
                "inner_selection": []}
        if search:
            for held in training:
                inner_train = sorted(set(training) - {held})
                inner_id = f"{prefix}-inner-{held}"
                fold["inner_selection"].append({
                    "id": inner_id, "training_matches": inner_train,
                    "assessment_matches": [held],
                    "prediction_requests": _requests(inner_train, [held], assessment, inner_id)})
        folds.append(fold)
    return {"schema": PROTOCOL_SCHEMA, "mode": "nested_search" if search else "nested_fixed",
            "development_only": True, "match_grouping": "all halves and scenarios stay together",
            "matches": ids, "historical_selection_matches": historical,
            "historical_reuse_caveat": HISTORICAL_REUSE_CAVEAT,
            "ordinary_oof_cache_is_sufficient": False, "folds": folds}


def build_holdout_plan(base_train, calibration, assessment, *, historical_selection_matches=()):
    """One disjoint base-training / calibration / untouched assessment plan."""
    base = _matches(base_train, "base training", nonempty=True)
    cal = _matches(calibration, "calibration", nonempty=True)
    test = _matches(assessment, "assessment", nonempty=True)
    if base & cal or base & test or cal & test:
        raise ValueError("holdout match groups overlap")
    return {"schema": PROTOCOL_SCHEMA, "mode": "holdout_fixed", "development_only": True,
            "match_grouping": "all halves and scenarios stay together",
            "matches": sorted(base | cal | test), "base_training_matches": sorted(base),
            "calibration_matches": sorted(cal), "assessment_matches": sorted(test),
            "historical_selection_matches": sorted(_matches(
                historical_selection_matches, "historical selection")),
            "historical_reuse_caveat": HISTORICAL_REUSE_CAVEAT,
            "ordinary_oof_cache_is_sufficient": False,
            "prediction_requests": [
                {"id": "holdout-calibration", "role": "calibration",
                 "predict_matches": sorted(cal), "train_matches": sorted(base),
                 "forbidden_matches": sorted(cal | test)},
                {"id": "holdout-assessment", "role": "assessment",
                 "predict_matches": sorted(test), "train_matches": sorted(base),
                 "forbidden_matches": sorted(cal | test)}]}


def load_frozen_protocol(path, expected=None):
    """Verify the entire canonical record and optional requested protocol."""
    path = Path(path)
    def no_duplicates(pairs):
        row = {}
        for key, value in pairs:
            if key in row:
                raise ValueError("frozen protocol contains duplicate JSON keys")
            row[key] = value
        return row
    record = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
    _object(record, ("sha256", "payload"), "frozen protocol")
    _hash(record["sha256"], "protocol")
    if canonical_sha256(record["payload"]) != record["sha256"]:
        raise ValueError("frozen protocol checksum differs")
    if expected is not None and canonical_sha256(expected) != record["sha256"]:
        raise ValueError("frozen protocol differs from requested protocol")
    return record["payload"]


def freeze_protocol(path, payload):
    """Atomically publish a new immutable protocol, or verify identical reuse.

    All writers serialize through an advisory lock on a stable sibling inode.
    Keep that lock file permanently, since removing it could let waiting writers
    lock different inodes. Publication uses a flushed sibling temporary file and
    atomic replacement, supported on local filesystems including exFAT. A crash
    releases the lock automatically. Readers see either no record or a complete
    record, and cooperating writers never replace an existing protocol.
    """
    path = Path(path)
    canonical = json.loads(_canonical_bytes(payload))
    encoded = _canonical_bytes({"sha256": canonical_sha256(canonical), "payload": canonical})
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if path.exists():
                return load_frozen_protocol(path, expected=canonical)
            descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(encoded + b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                return load_frozen_protocol(path, expected=canonical)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

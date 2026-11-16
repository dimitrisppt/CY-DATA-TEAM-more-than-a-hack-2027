"""Matched, public-only continuation experiments for short possession recovery.

The original v6 model and decoder stay frozen. Each ablation starts from the
same authenticated train-four state, runs a fixed number of extra updates,
and is assessed on the two excluded matches. No validation-driven stopping
or configuration search is performed. Historically reused matches remain
development evidence, even with direct training ancestry exclusions.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments import campaign_evaluation, campaign_training
from experiments.campaign_decode import decode_arrays
from experiments.campaign_protocol import WEIGHTS, Budget, BudgetExceeded, read_record
from experiments.coarse_context import blend_coarse_context
from experiments.coarse_stability import _bagged_loaded
from experiments.ema_stability import _verify_input_cache
from experiments.robustness import _source_hashes
from experiments.spell_evidence_audit import spell_counts
from experiments.timing_cache import _check_hash, _fit_component, _required
from experiments.timing_validation import (
    canonical_sha256,
    freeze_protocol,
    validate_prediction_provenance,
)
from possession.predict import load_model_checkpoint
from possession.prepared import verify_prepared
from possession.storage import sha256_file, write_json
from possession.training_session import HalfCache
from scripts.evaluate_reference import _score_rows

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = Path('/Volumes/T7/Data/more-than-a-hack-2027/artifacts')
VARIANTS = ('control', 'spell_balanced', 'masked_teacher')
SEEDS = (8101, 8102)
BATCHES = ('batch_1', 'batch_2')
SETTINGS = {
    'start_steps': 7200, 'total_extra_steps': 1200, 'lr_scale': .2,
    'spell_weight': .25, 'short_sampling_probability': .25,
    'distill_weight': .25, 'temperature': 2., 'confidence': .8,
}
NEW_SOURCES = (
    'experiments/possession_objectives.py', 'experiments/refinement_training.py',
    'experiments/possession_refinement.py', 'experiments/spell_evidence_audit.py',
)
CAVEAT = (
    'All seven public matches have prior development use. Whole-match fit exclusions '
    'prevent direct training leakage, but do not remove historical selection bias. '
    'Two mask batches and two training seeds are paired robustness checks, not '
    'independent matches. No challenge possession labels are read or used.'
)


def validate_folds(folds):
    """Require the fixed disjoint assessment pairs, with their exact complements."""
    if len(folds) != 3:
        raise ValueError('exactly three folds required')
    universe = {m for fold in folds for m in fold['validation']}
    if len(universe) != 6 or 'DFL-MAT-J03WMX' in universe:
        raise ValueError('six inner matches required, with WMX excluded')
    seen = set()
    for fold in folds:
        train, valid = fold['train'], fold['validation']
        if (len(train) != 4 or len(set(train)) != 4 or len(valid) != 2
                or len(set(valid)) != 2 or seen.intersection(valid)
                or set(train) != universe - set(valid)):
            raise ValueError('fold training ancestry or assessment partition differs')
        seen.update(valid)


def preflight(artifacts_root, output_dir):
    """Authenticate frozen parents, masks, coarse context and prepared public data."""
    root, output = Path(artifacts_root).resolve(), Path(output_dir).resolve()
    if output == root or root.is_relative_to(output):
        raise ValueError('output must not contain the source artifact root')
    for name in ('overnight-joint-v1', 'ema-stability-v1', 'coarse-stability-v1',
                 'ensemble-bagging-v1', 'timing-calibration-v1'):
        if output.is_relative_to(root / name):
            raise ValueError('output must be separate from frozen studies')
    campaign = _required(root / 'overnight-joint-v1/protocol.json')
    coarse = _required(root / 'coarse-stability-v1/protocol.json')
    ema = _required(root / 'ema-stability-v1/protocol.json')
    bag = _required(root / 'ensemble-bagging-v1/protocol.json')
    cache_path = root / 'timing-calibration-v1/cache/manifest.json'
    cache = _required(cache_path)
    if (cache.get('schema') != 'public_dfl_timing_cache_v1'
            or cache.get('target_ground_truth_used') is not False
            or cache.get('source_protocol_sha256') != {
                'coarse-stability-v1': canonical_sha256(coarse),
                'ema-stability-v1': canonical_sha256(ema),
                'ensemble-bagging-v1': canonical_sha256(bag)}):
        raise ValueError('rich cache source protocols differ')
    source = _source_hashes()
    if any(source.get(name) != digest for name, digest in coarse['source_sha256'].items()):
        raise ValueError('pinned v6 source differs')
    selection, folds = coarse['candidate_selection'], coarse['inner_folds']
    validate_folds(folds)
    if (folds != campaign['fold']['inner'] or folds != cache['inner_folds']
            or not cache.get('exact_v6_parity') or cache['decoder'] != selection['decoder']):
        raise ValueError('frozen cache, decoder or folds differ')
    data_dir = Path(bag['data_dir']).resolve()
    prepared = verify_prepared(data_dir, sorted(data_dir.glob('DFL-MAT-*_half*.npz')))
    if prepared != campaign['prepared']:
        raise ValueError('public prepared data differs')
    matches = {m for f in folds for m in f['validation']}
    inputs, input_sha = {}, {}
    for batch in BATCHES:
        result = _required(root / 'ensemble-bagging-v1' / batch / 'result.json')
        input_sha[batch] = result['inner_inputs_sha256']
        inputs[batch], _ = _verify_input_cache(
            root / 'ensemble-bagging-v1' / batch / 'inner', input_sha[batch], matches)
    parents = []
    coarse_parents = []
    components = {index: {} for index in range(len(folds))}
    for index, fold in enumerate(folds):
        for seed in SEEDS:
            for member in selection['members']:
                fit, fit_path, model_path, component = _fit_component(
                    root, 'inner', seed, index, fold, member['spec'], member['id'],
                    canonical_sha256(campaign), fine=True)
                components[index][component.name] = asdict(component)
                state = (root / 'overnight-joint-v1/fits/inner' / member['id']
                         / f'seed_{seed}' / f'fold_{index}' / 'state_007200.pt')
                parents.append({
                    'fold': index, 'seed': seed, 'name': member['id'], 'spec': member['spec'],
                    'parent_identity': fit['identity'], 'train_matches': fold['train'],
                    'validation_matches': fold['validation'], 'state_path': str(state),
                    'state_sha256': fit['inputs']['end_state_sha256'],
                    'teacher_path': str(model_path), 'teacher_sha256': fit['checkpoint_sha256'],
                    'teacher_fit_sha256': sha256_file(fit_path),
                })
            fit, fit_path, _, component = _fit_component(
                root, 'inner', seed, index, fold, coarse['model_spec'],
                'coarse_player_only', canonical_sha256(coarse), fine=False)
            components[index][component.name] = asdict(component)
            coarse_parents.append({'fold': index, 'seed': seed,
                                   'fit_sha256': sha256_file(fit_path),
                                   'model_sha256': fit['checkpoint_sha256']})
    rows = [row for row in cache['records'] if row['scope'] == 'inner']
    expected = {(b, m, p, s) for b in BATCHES for m in matches for p in (1, 2) for s in WEIGHTS}
    keys = {(r['batch'], r['match'], r['period'], r['scenario']) for r in rows}
    if len(rows) != len(keys) or keys != expected:
        raise ValueError('rich public cache grid differs')
    input_index = {(b, r['match'], r['period'], r['scenario']): r
                   for b, batch_rows in inputs.items() for r in batch_rows}
    for row in rows:
        key = (row['batch'], row['match'], row['period'], row['scenario'])
        if row['source_input_sha256'] != input_index[key]['input_sha256']:
            raise ValueError('rich evidence and inference input masks differ')
        fold = row['fold']
        if fold not in components or row['match'] not in folds[fold]['validation']:
            raise ValueError('rich cache assessment fold differs')
        actual_components = {c['name']: c for c in row['provenance']['components']}
        for name, component in components[fold].items():
            if actual_components.get(name) != component:
                raise ValueError('rich cache fitted ancestor differs')
        validate_prediction_provenance(
            row['provenance'], outer_assessment_matches=folds[fold]['validation'],
            allowed_matches=matches | {'DFL-MAT-J03WMX'},
            expected_components=set(components[fold]) | {'fixed_features_reference_blend'})
        path = Path(row['path']).resolve()
        if not path.is_relative_to(cache_path.parent):
            raise ValueError('rich cache path outside authenticated public root')
        _check_hash(path, row['sha256'])
    protocol = {
        'schema': 'possession_refinement_protocol_v1', 'development_only': True,
        'historical_reuse_caveat': CAVEAT, 'data_dir': str(data_dir),
        'prepared': prepared, 'selection': selection, 'folds': folds,
        'variants': list(VARIANTS), 'seeds': list(SEEDS), 'batches': list(BATCHES),
        'settings': SETTINGS, 'tail_ema': {'last_updates': 200, 'decay': .99},
        'scenario_weights': WEIGHTS, 'input_signatures': input_sha,
        'cache_manifest_sha256': sha256_file(cache_path),
        'parents': parents, 'coarse_parents': coarse_parents,
        'pinned_source_sha256': coarse['source_sha256'],
        'experiment_source_sha256': {name: sha256_file(REPO / name) for name in NEW_SOURCES},
        'resources': {'cpu_threads': 4, 'workers': 1, 'cache_bytes': 1_000_000_000},
        'evaluation': {'automatic_promotion': False, 'guard': 'WMX not used in this screen',
                       'objective_gain_over_v6_gt': .002,
                       'objective_gain_over_control_gt': .002,
                       'improved_matches_at_least': 4,
                       'missing_gain_gt': 0., 'each_batch_gain_gt': 0.,
                       'scenario_loss_at_most': .01,
                       'positive_screen_action': 'fresh confirmation and separate WMX guard'},
    }
    freeze_protocol(output / 'protocol.json', protocol)
    return protocol, inputs, rows


def _fit_id(parent, variant, protocol):
    return {'protocol_sha256': canonical_sha256(protocol), 'variant': variant,
            'fold': parent['fold'], 'seed': parent['seed'], 'name': parent['name'],
            'train_matches': parent['train_matches'],
            'validation_matches': parent['validation_matches'],
            'parent_state_sha256': parent['state_sha256'],
            'teacher_sha256': parent['teacher_sha256'] if variant == 'masked_teacher' else None}


def _train_one(parent, variant, protocol, output, cache, budget):
    from experiments.refinement_training import (
        configure_refinement,
        export_refinement_checkpoint,
        load_refinement_state,
        save_refinement_state,
    )

    identity = _fit_id(parent, variant, protocol)
    directory = (output / 'fits' / variant / f"fold_{parent['fold']}"
                 / f"seed_{parent['seed']}" / parent['name'])
    path = directory / 'model.pt'
    fit_path = directory / 'fit.json'
    old = read_record(fit_path)
    if old is not None:
        if old['identity'] != identity or old['steps'] != 8400:
            raise ValueError('existing refinement fit identity differs')
        _check_hash(path, old['checkpoint_sha256'])
        return old, path
    budget.check()
    started = time.monotonic()
    session = campaign_training.build_session(
        protocol['data_dir'], parent['spec'], parent['train_matches'],
        parent['parent_identity']['seed'], cache)
    _check_hash(parent['state_path'], parent['state_sha256'])
    campaign_training.load_training_state(parent['state_path'], session, parent['parent_identity'])
    teacher = None
    if variant == 'masked_teacher':
        _check_hash(parent['teacher_path'], parent['teacher_sha256'])
        teacher, _ = load_model_checkpoint(parent['teacher_path'])
    session = configure_refinement(session, variant=variant, teacher=teacher, **SETTINGS)
    states = sorted(p for p in directory.glob('state_*.pt')
                    if p.with_suffix('.pt.sha256.json').is_file())
    if states:
        load_refinement_state(states[-1], session, identity)
    for step in range(session.steps + 200, 8401, 200):
        budget.status('training', variant=variant, fold=parent['fold'], seed=parent['seed'],
                      member=parent['name'], step=session.steps, end_step=8400)
        budget.check()
        session.advance_to(step)
        save_refinement_state(directory / f'state_{step:06d}.pt', session, identity)
        print(json.dumps({'phase': 'training', 'variant': variant, 'fold': parent['fold'],
                          'seed': parent['seed'], 'member': parent['name'], 'step': step,
                          'seconds': round(time.monotonic() - started, 2)}), flush=True)
    if session.steps != 8400:
        raise ValueError('refinement endpoint differs')
    export_refinement_checkpoint(path, session, identity, use_tail_ema=True)
    model, payload = load_model_checkpoint(path)
    del model
    config = payload['training_config']
    if config['train_matches'] != parent['train_matches'] or config['steps'] != 8400:
        raise ValueError('exported refinement checkpoint ancestry differs')
    fit = {'identity': identity, 'steps': session.steps,
           'checkpoint_sha256': sha256_file(path),
           'invocation_seconds': time.monotonic() - started,
           'progress': session._progress()}
    freeze_protocol(fit_path, fit)
    del session, teacher
    gc.collect()
    return fit, path


def _predict_one(parent, variant, protocol, output, model_path, fit, inputs, budget):
    records = []
    model, _ = load_model_checkpoint(model_path)
    for batch in BATCHES:
        selected = [row for row in inputs[batch] if row['match'] in parent['validation_matches']]
        if any(row['match'] in parent['train_matches'] for row in selected):
            raise ValueError('prediction match appears in training ancestry')
        identity = {**fit['identity'], 'batch': batch,
                    'checkpoint_sha256': fit['checkpoint_sha256'],
                    'input_manifest_sha256': protocol['input_signatures'][batch]}
        budget.status('predicting', variant=variant, fold=parent['fold'], seed=parent['seed'],
                      member=parent['name'], batch=batch)
        rows = campaign_evaluation.predict_records(
            model, selected, output / 'inference' / batch, identity,
            progress_callback=budget.check)
        records.extend({**row, 'batch': batch, 'variant': variant,
                        'seed': parent['seed'], 'member': parent['name']} for row in rows)
    del model
    gc.collect()
    return records


def aggregate_scores(scores):
    """Average mask batches once per match, never count masks as new matches."""
    if set(scores) != set(BATCHES):
        raise ValueError('both fixed mask batches required')
    matches = sorted(scores[BATCHES[0]]['by_match'])
    if len(matches) != 6 or any(sorted(row['by_match']) != matches for row in scores.values()):
        raise ValueError('the same six whole matches are required')
    by_match = {match: {scenario: float(np.mean([
        scores[batch]['by_match'][match][scenario]['spell_f1_mean'] for batch in BATCHES
    ])) for scenario in WEIGHTS} for match in matches}
    means = {scenario: float(np.mean([row[scenario] for row in by_match.values()]))
             for scenario in WEIGHTS}
    return {'objective': float(sum(WEIGHTS[s] * means[s] for s in WEIGHTS)),
            'missing': float(sum(WEIGHTS[s] * means[s] for s in WEIGHTS if s != 'clean') / .5),
            'means': means, 'by_match': by_match,
            'by_batch': {b: scores[b]['objective'] for b in BATCHES}}


def comparison_gate(baseline, control, candidate):
    """A screen can justify confirmation, never deployment or a test-set claim."""
    def delta_to(reference):
        deltas = {m: sum(WEIGHTS[s] * (candidate['by_match'][m][s]
                        - reference['by_match'][m][s]) for s in WEIGHTS)
                  for m in reference['by_match']}
        return {'objective': candidate['objective'] - reference['objective'],
                'missing': candidate['missing'] - reference['missing'],
                'by_match': deltas,
                'by_scenario': {s: candidate['means'][s] - reference['means'][s]
                                for s in WEIGHTS},
                'by_batch': {b: candidate['by_batch'][b] - reference['by_batch'][b]
                             for b in BATCHES}}
    to_v6, to_control = delta_to(baseline), delta_to(control)
    checks = {}
    for label, delta in (('v6', to_v6), ('control', to_control)):
        checks.update({f'gain_over_{label}': delta['objective'] > .002,
                       f'missing_over_{label}': delta['missing'] > 0.,
                       f'breadth_over_{label}': sum(v > 0. for v in delta['by_match'].values()) >= 4,
                       f'batches_over_{label}': all(v > 0. for v in delta['by_batch'].values()),
                       f'protect_scenarios_over_{label}':
                           all(v >= -.01 for v in delta['by_scenario'].values())})
    return {'vs_v6': to_v6, 'vs_control': to_control, 'checks': checks,
            'confirmation_eligible': all(checks.values()), 'automatic_promotion': False}


def evaluate_predictions(protocol, rich_rows, predictions, output):
    """Export a portable label reference and score exact full halves officially."""
    index = {}
    for row in predictions:
        key = (row['variant'], row['batch'], row['seed'], row['member'],
               row['match'], row['period'], row['scenario'])
        if key in index:
            raise ValueError('duplicate refinement prediction')
        index[key] = row
    members = [member['id'] for member in protocol['selection']['members']]
    expected = {(v, r['batch'], seed, m, r['match'], r['period'], r['scenario'])
                for r in rich_rows for v in VARIANTS for seed in SEEDS for m in members}
    if set(index) != expected:
        raise ValueError('incomplete refinement prediction grid')
    labels, references = {}, []
    for source in rich_rows:
        _check_hash(source['path'], source['sha256'])
        with np.load(source['path'], allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        batch, match, period, scenario = (source[k] for k in ('batch', 'match', 'period', 'scenario'))
        gt_key = f'gt_{match}_p{period}'
        if gt_key in labels and not np.array_equal(labels[gt_key], arrays['gt']):
            raise ValueError('public ground truth differs across scenarios')
        labels[gt_key] = arrays['gt'].astype(np.uint8)
        if not np.array_equal(decode_arrays(arrays, protocol['selection']['decoder']),
                              arrays['v6_labels']):
            raise ValueError('frozen decoder parity differs before assessment')
        keys = {'v6': f'v6_{batch}_{match}_{scenario}_p{period}'}
        labels[keys['v6']] = arrays['v6_labels'].astype(np.uint8)
        for variant in VARIANTS:
            by_seed = {}
            for seed in SEEDS:
                bundles = []
                for name in members:
                    row = index[(variant, batch, seed, name, match, period, scenario)]
                    if row['input_sha256'] != source['source_input_sha256']:
                        raise ValueError('prediction input mask differs from rich evidence')
                    _check_hash(row['prediction_path'], row['prediction_sha256'])
                    with np.load(row['prediction_path'], allow_pickle=False) as saved:
                        neural = {k: saved[k] for k in saved.files}
                    bundles.append([{**row, 'arrays': {**arrays, **neural}}])
                by_seed[seed] = bundles
            fine = _bagged_loaded(by_seed, protocol['selection'])[0]['arrays']
            mixed = blend_coarse_context(fine, arrays['coarse_live'], arrays['coarse_home'])
            keys[variant] = f'{variant}_{batch}_{match}_{scenario}_p{period}'
            labels[keys[variant]] = np.asarray(
                decode_arrays(mixed, protocol['selection']['decoder']), dtype=np.uint8)
        references.append({'batch': batch, 'scope': 'inner', 'match': match, 'period': period,
                           'scenario': scenario, 'n_frames': len(arrays['gt']),
                           'ground_truth_key': gt_key, 'prediction_keys': keys})
    manifest = {'schema': 'public_possession_refinement_reference_v1',
                'protocol_sha256': canonical_sha256(protocol),
                'scenario_weights': WEIGHTS, 'scenario_order': list(WEIGHTS),
                'records': references, 'historical_reuse_caveat': CAVEAT}
    reference_dir = output / 'reference'
    reference_dir.mkdir(parents=True, exist_ok=True)
    path = reference_dir / 'labels.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(labels) or any(
                    not np.array_equal(saved[k], value) for k, value in labels.items()):
                raise ValueError('completed portable labels differ')
    else:
        campaign_evaluation._atomic_npz(path, labels, compressed=True)
    manifest['labels_sha256'] = sha256_file(path)
    freeze_protocol(reference_dir / 'manifest.json', manifest)
    result = score_reference(labels, manifest)
    freeze_protocol(output / 'result.json', result)
    return result


def score_reference(labels, manifest):
    """Score only decoded labels, without models, tracking files or training data."""
    raw = {v: {b: _score_rows(labels, manifest, v, b, 'inner') for b in BATCHES}
           for v in ('v6', *VARIANTS)}
    summaries = {v: aggregate_scores(rows) for v, rows in raw.items()}
    gates = {v: comparison_gate(summaries['v6'], summaries['control'], summaries[v])
             for v in VARIANTS if v != 'control'}
    diagnostics = {}
    for variant in ('v6', *VARIANTS):
        diagnostics[variant] = {}
        for batch in BATCHES:
            diagnostics[variant][batch] = {}
            for scenario in WEIGHTS:
                selected = [r for r in manifest['records']
                            if r['batch'] == batch and r['scenario'] == scenario]
                counts, bins = {}, {}
                for row in selected:
                    measured = spell_counts(labels[row['ground_truth_key']],
                                            labels[row['prediction_keys'][variant]])
                    for key, value in measured['counts'].items():
                        counts[key] = counts.get(key, 0) + value
                    for name, values in measured['duration_bins'].items():
                        target = bins.setdefault(name, {})
                        for key, value in values.items():
                            if key != 'recall_at_0_5':
                                target[key] = target.get(key, 0) + value
                for values in bins.values():
                    values['recall_at_0_5'] = (values['matched_at_0_5'] / values['gt']
                                               if values['gt'] else 0.)
                diagnostics[variant][batch][scenario] = {
                    'counts': counts, 'duration_bins': bins}
    return {'protocol_sha256': manifest['protocol_sha256'], 'scores': raw,
              'summary': summaries, 'gates': gates, 'decision': 'retain_v6',
              'confirmation_candidates': [v for v, g in gates.items() if g['confirmation_eligible']],
              'spell_diagnostics': diagnostics, 'historical_reuse_caveat': CAVEAT,
              'challenge_submission_modified': False}


def evaluate_reference(directory):
    """Reproduce an authenticated portable refinement reference."""
    directory = Path(directory)
    manifest = _required(directory / 'manifest.json')
    if (manifest.get('schema') != 'public_possession_refinement_reference_v1'
            or manifest.get('scenario_weights') != WEIGHTS):
        raise ValueError('unsupported refinement label reference')
    _check_hash(directory / 'labels.npz', manifest['labels_sha256'])
    with np.load(directory / 'labels.npz', allow_pickle=False) as labels:
        return score_reference(labels, manifest)


def run(artifacts_root, output_dir, *, hours=8., plan_only=False):
    import torch

    if not np.isfinite(hours) or not 0 < hours <= 12:
        raise ValueError('compute budget must be between zero and twelve hours')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    output = Path(output_dir).resolve()
    protocol, inputs, rich = preflight(artifacts_root, output)
    print(json.dumps({'phase': 'preflight_complete', 'protocol_sha256': canonical_sha256(protocol),
                      'fits': len(protocol['parents']) * len(VARIANTS),
                      'assessment_halves': len(rich), 'plan_only': plan_only}), flush=True)
    if plan_only:
        return protocol
    budget = Budget(output, hours)
    cache = HalfCache(protocol['data_dir'], max_bytes=1_000_000_000)
    predictions = []
    try:
        for parent in protocol['parents']:
            for variant in VARIANTS:
                fit, model_path = _train_one(parent, variant, protocol, output, cache, budget)
                predictions.extend(_predict_one(
                    parent, variant, protocol, output, model_path, fit, inputs, budget))
        budget.check()
        budget.status('scoring_all_complete_variants')
        freeze_protocol(output / 'predictions.json', {'records': predictions})
        result = evaluate_predictions(protocol, rich, predictions, output)
        write_json(output / 'status.json', {'status': 'complete', 'pid': os.getpid(),
                   'finished_unix': time.time(), 'decision': result['decision'],
                   'confirmation_candidates': result['confirmation_candidates'],
                   'summary': result['summary']})
        print(json.dumps({'phase': 'complete', 'decision': result['decision'],
                          'summary': result['summary']}), flush=True)
        return result
    except BudgetExceeded:
        write_json(output / 'status.json', {'status': 'budget_exhausted_resumable',
                   'pid': os.getpid(), 'clock': budget.clock, 'finished_unix': time.time()})
        raise
    except Exception as exc:
        write_json(output / 'status.json', {'status': 'failed_resumable',
                   'pid': os.getpid(), 'error': f'{type(exc).__name__}: {exc}',
                   'updated_unix': time.time()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-root', type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument('--output-dir', type=Path,
                        default=DEFAULT_ARTIFACTS / 'possession-refinement-v1')
    parser.add_argument('--hours', type=float, default=8.)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--score-reference', type=Path,
                        help='recompute saved label metrics without training or raw data')
    args = parser.parse_args()
    if args.score_reference is not None:
        print(json.dumps(evaluate_reference(args.score_reference), indent=2))
    else:
        run(args.artifacts_root, args.output_dir, hours=args.hours, plan_only=args.plan_only)


if __name__ == '__main__':
    main()

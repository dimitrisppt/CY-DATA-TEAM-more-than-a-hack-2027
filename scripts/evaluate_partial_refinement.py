"""Describe a stopped refinement study using completed, saved predictions only.

This post-stop analysis is not the planned complete cross-validation result.
It performs no training, inference, tuning, promotion or study-status changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.campaign_decode import decode_arrays
from experiments.campaign_evaluation import PREDICTION_SCHEMA, _atomic_npz
from experiments.campaign_protocol import WEIGHTS
from experiments.coarse_context import blend_coarse_context
from experiments.coarse_stability import _bagged_loaded
from experiments.ensemble_polish_search import blend_loaded
from experiments.possession_refinement import _fit_id
from experiments.spell_evidence_audit import spell_counts
from experiments.timing_cache import _check_hash, _fine_arrays, _required
from experiments.timing_validation import canonical_sha256
from possession.storage import sha256_file, write_json
from scripts.evaluate_reference import _score_rows

DEFAULT_ROOT = Path('/Volumes/T7/Data/more-than-a-hack-2027/artifacts')
DEFAULT_OUTPUT = Path('outputs/possession-refinement/partial-comparison')
GROUPS = {
    'matched_single_fine_seed': {
        'seeds': [8101], 'variants': ['baseline', 'control', 'spell_balanced', 'masked_teacher'],
    },
    'completed_two_fine_seeds': {
        'seeds': [8101, 8102], 'variants': ['baseline', 'control', 'spell_balanced'],
    },
}


def summarize(scores):
    batches = sorted(scores)
    matches = sorted(scores[batches[0]]['by_match'])
    if any(sorted(value['by_match']) != matches for value in scores.values()):
        raise ValueError('paired assessment matches differ')
    by_match = {m: {s: float(np.mean([
        scores[b]['by_match'][m][s]['spell_f1_mean'] for b in batches
    ])) for s in WEIGHTS} for m in matches}
    means = {s: float(np.mean([row[s] for row in by_match.values()])) for s in WEIGHTS}
    return {'objective': sum(WEIGHTS[s] * means[s] for s in WEIGHTS),
            'missing': sum(WEIGHTS[s] * means[s] for s in WEIGHTS if s != 'clean') / .5,
            'means': means, 'by_match': by_match,
            'match_objectives': {m: sum(WEIGHTS[s] * row[s] for s in WEIGHTS)
                                 for m, row in by_match.items()},
            'batch_objectives': {b: scores[b]['objective'] for b in batches}}


def run(root=DEFAULT_ROOT, output=DEFAULT_OUTPUT):
    root, output = Path(root).resolve(), Path(output).resolve()
    study = root / 'possession-refinement-v1'
    if output == study or output.is_relative_to(study):
        raise ValueError('partial analysis must be outside the frozen study directory')
    if json.loads((study / 'status.json').read_text())['status'] != 'stopped_by_user':
        raise ValueError('this analysis is only for the explicitly stopped study')
    protocol = _required(study / 'protocol.json')
    for name, digest in protocol['experiment_source_sha256'].items():
        _check_hash(Path(__file__).resolve().parents[1] / name, digest)
    cache_path = root / 'timing-calibration-v1/cache/manifest.json'
    _check_hash(cache_path, protocol['cache_manifest_sha256'])
    cache = _required(cache_path)
    sources = [r for r in cache['records'] if r['scope'] == 'inner' and r['fold'] == 0]
    fold = protocol['folds'][0]
    members = [m['id'] for m in protocol['selection']['members']]
    batches = protocol['batches']
    grid = {(b, m, p, s) for b in batches for m in fold['validation']
            for p in (1, 2) for s in WEIGHTS}
    if {(r['batch'], r['match'], r['period'], r['scenario']) for r in sources} != grid:
        raise ValueError('partial assessment grid differs')
    parents = {(p['seed'], p['name']): p for p in protocol['parents'] if p['fold'] == 0}
    fits, neural_index, metadata_hashes = {}, {}, {}
    for path in sorted((study / 'fits').rglob('fit.json')):
        fit = _required(path)
        identity = fit['identity']
        variant, seed, name = (identity[k] for k in ('variant', 'seed', 'name'))
        if identity['fold'] != 0:
            continue
        parent = parents[(seed, name)]
        if identity != _fit_id(parent, variant, protocol) or fit['steps'] != 8400:
            raise ValueError('completed fit identity or budget differs')
        _check_hash(path.parent / 'model.pt', fit['checkpoint_sha256'])
        if (set(parent['train_matches']) & set(fold['validation'])
                or fit['progress']['refinement']['tail_ema_updates'] != 200):
            raise ValueError('fit ancestry or final EMA is invalid')
        fits[(variant, seed, name)] = fit
        metadata_hashes[str(path)] = sha256_file(path)
    for batch in batches:
        for path in (study / 'inference' / batch).rglob('*.json'):
            if path.name.startswith('._'):
                continue
            meta = json.loads(path.read_text())
            mid = meta['identity']['model_identity']
            key = (mid['variant'], mid['seed'], mid['name'])
            if key not in fits:
                continue
            expected = {**fits[key]['identity'], 'batch': batch,
                        'checkpoint_sha256': fits[key]['checkpoint_sha256'],
                        'input_manifest_sha256': protocol['input_signatures'][batch]}
            if mid != expected:
                raise ValueError('prediction fit identity differs')
            ident = meta['identity']
            record_key = (*key, batch, ident['match'], ident['period'], ident['scenario'])
            if record_key in neural_index:
                raise ValueError('duplicate saved prediction')
            neural_index[record_key] = (path.with_suffix('.npz'), meta)
    baseline = {}
    ema_protocol_sha = canonical_sha256(_required(root / 'ema-stability-v1/protocol.json'))
    for batch in batches:
        for seed in (8101, 8102):
            for name in members:
                parent = parents[(seed, name)]
                fit_path = Path(parent['teacher_path']).with_name('fit.json')
                _check_hash(fit_path, parent['teacher_fit_sha256'])
                _check_hash(parent['teacher_path'], parent['teacher_sha256'])
                fit = _required(fit_path)
                if fit['identity'] != parent['parent_identity']:
                    raise ValueError('baseline parent ancestry differs')
                path = (root / 'ema-stability-v1/predictions' / batch / 'inner'
                        / f'seed_{seed}' / name / 'fold_00.json')
                manifest = _required(path)
                expected = {
                    'protocol_sha256': ema_protocol_sha,
                    'ema_fit_sha256': parent['teacher_fit_sha256'],
                    'ema_sha256': fit['ema_sha256'],
                    'ema_checkpoint_sha256': parent['teacher_sha256'],
                    'raw_checkpoint_sha256': fit['inputs']['baseline_checkpoint_sha256'],
                    'scope': 'inner', 'seed': seed, 'fold': 0,
                    'train_matches': fold['train'], 'validation_matches': fold['validation'],
                    'input_manifest_sha256': protocol['input_signatures'][batch],
                }
                if manifest['identity'] != expected:
                    raise ValueError('baseline prediction ancestry differs')
                baseline[(batch, seed, name)] = (expected, {
                    (r['match'], r['period'], r['scenario']): r for r in manifest['records']})
                metadata_hashes[str(path)] = sha256_file(path)
    labels, references = {}, {name: [] for name in GROUPS}
    for source in sources:
        _check_hash(source['path'], source['sha256'])
        with np.load(source['path'], allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        batch, match, period, scenario = (source[k] for k in ('batch', 'match', 'period', 'scenario'))
        expected_input = source['source_input_sha256']
        identity_row = {'match': match, 'period': period, 'scenario': scenario,
                        'input_sha256': expected_input}
        prediction_arrays = {}
        for seed in (8101, 8102):
            for name in members:
                expected, indexed = baseline[(batch, seed, name)]
                record = indexed[(match, period, scenario)]
                bound_source = {**record, 'input_sha256': expected_input,
                                'input_identity': protocol['input_signatures'][batch]}
                prediction_arrays[('baseline', seed, name)] = _fine_arrays(
                    record, bound_source, expected, root)
                for variant in ('control', 'spell_balanced', 'masked_teacher'):
                    if variant == 'masked_teacher' and seed != 8101:
                        continue
                    path, meta = neural_index[(variant, seed, name, batch, match, period, scenario)]
                    expected_meta = {'schema': PREDICTION_SCHEMA, 'input_sha256': expected_input,
                                     'model_identity': meta['identity']['model_identity'],
                                     'center_size': 2048, 'match': match, 'period': period,
                                     'scenario': scenario}
                    if meta['identity'] != expected_meta or meta['n_frames'] != len(arrays['gt']):
                        raise ValueError('prediction input mask or grid differs')
                    _check_hash(path, meta['prediction_sha256'])
                    with np.load(path, allow_pickle=False) as saved:
                        if set(saved.files) != {'neural_live', 'neural_home', 'neural_boundary'}:
                            raise ValueError('unexpected saved prediction fields')
                        prediction_arrays[(variant, seed, name)] = {k: saved[k] for k in saved.files}
                    metadata_hashes[str(path.with_suffix('.json'))] = sha256_file(path.with_suffix('.json'))
        gt_key = f'gt_{match}_{period}'
        if gt_key in labels and not np.array_equal(labels[gt_key], arrays['gt']):
            raise ValueError('ground truth differs across paired masks')
        labels[gt_key] = arrays['gt'].astype(np.uint8)
        for group, configuration in GROUPS.items():
            keys = {}
            for variant in configuration['variants']:
                by_seed = {seed: [[{**identity_row, 'arrays': {
                    **arrays, **prediction_arrays[(variant, seed, name)]
                }}] for name in members] for seed in configuration['seeds']}
                if len(by_seed) == 2:
                    fine = _bagged_loaded(by_seed, protocol['selection'])[0]['arrays']
                else:
                    member_rows = by_seed[8101]
                    fine = blend_loaded(member_rows[0], member_rows[1],
                                        protocol['selection']['blend_weight'])[0]['arrays']
                mixed = blend_coarse_context(fine, arrays['coarse_live'], arrays['coarse_home'])
                prediction = np.asarray(decode_arrays(mixed, protocol['selection']['decoder']),
                                        dtype=np.uint8)
                if (group == 'completed_two_fine_seeds' and variant == 'baseline'
                        and not np.array_equal(prediction, arrays['v6_labels'])):
                    raise ValueError('reconstructed full v6 differs from frozen labels')
                key = f'{group}_{variant}_{batch}_{match}_{period}_{scenario}'
                labels[key], keys[variant] = prediction, key
            references[group].append({**identity_row, 'batch': batch, 'scope': 'inner',
                'ground_truth_key': gt_key, 'prediction_keys': keys, 'n_frames': len(arrays['gt'])})
    results = {}
    for group, config in GROUPS.items():
        manifest = {'records': references[group], 'scenario_weights': WEIGHTS}
        raw = {v: {b: _score_rows(labels, manifest, v, b, 'inner') for b in batches}
               for v in config['variants']}
        diagnostics = {v: {s: [spell_counts(labels[r['ground_truth_key']],
                            labels[r['prediction_keys'][v]]) for r in references[group]
                            if r['batch'] == 'batch_1' and r['scenario'] == s]
                           for s in WEIGHTS} for v in config['variants']}
        results[group] = {'configuration': config, 'scores': raw,
                          'summary': {v: summarize(score) for v, score in raw.items()},
                          'spell_diagnostics_batch_1': diagnostics}
    output.mkdir(parents=True, exist_ok=True)
    _atomic_npz(output / 'labels.npz', labels, compressed=True)
    write_json(output / 'manifest.json', {'schema': 'partial_refinement_reference_v1',
        'references': references, 'groups': GROUPS, 'scenario_weights': WEIGHTS,
        'labels_sha256': sha256_file(output / 'labels.npz')})
    report = {'schema': 'partial_refinement_analysis_v1',
        'original_protocol_sha256': canonical_sha256(protocol),
        'analysis_source_sha256': sha256_file(__file__), 'assessment_matches': fold['validation'],
        'completed_fits': len(fits), 'groups': results, 'metadata_sha256': metadata_hashes,
        'training_restarted': False, 'inference_run': False, 'promotion_evaluated': False,
        'limitations': ['Post-stop descriptive analysis of only the first of three folds.',
            'The assessment contains two historically reused matches, not an independent test.',
            'The one-fine-seed baseline is not the complete delivered v6 ensemble.',
            'Both frozen coarse seeds, decoder, blending and mask batches remain identical.',
            'The incomplete second teacher attention fit is excluded, not replaced or retrained.',
            'No full six-match result or advancement decision is available.']}
    write_json(output / 'result.json', report)
    print(json.dumps({group: value['summary'] for group, value in results.items()}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.artifacts_root, args.output_dir)


if __name__ == '__main__':
    main()

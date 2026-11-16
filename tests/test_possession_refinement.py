"""Synthetic orchestration checks, no model fitting or external datasets."""
import copy
import json
import sys

import numpy as np
import pytest

from experiments import possession_refinement as runner
from experiments.campaign_decode import baseline_decoder, decode_arrays
from experiments.campaign_protocol import read_record
from possession.storage import sha256_file

MATCHES = tuple(f'DFL-MAT-SYNTH{i}' for i in range(6))


def _folds():
    return [{'train': [m for m in MATCHES if m not in MATCHES[i:i + 2]],
             'validation': list(MATCHES[i:i + 2])} for i in (0, 2, 4)]


def _raw_scores(first=.5, second=.5):
    return {batch: {'objective': value,
                    'by_match': {m: {s: {'spell_f1_mean': value} for s in runner.WEIGHTS}
                                 for m in MATCHES}}
            for batch, value in zip(runner.BATCHES, (first, second), strict=True)}


def test_folds_exclude_entire_assessment_pairs_and_wmx():
    runner.validate_folds(_folds())
    for mutation in ('train_leak', 'repeated_assessment', 'wmx', 'duplicate_train'):
        folds = _folds()
        if mutation == 'train_leak':
            folds[0]['train'][0] = folds[0]['validation'][0]
        elif mutation == 'repeated_assessment':
            folds[1]['validation'][0] = folds[0]['validation'][0]
        elif mutation == 'wmx':
            for fold in folds:
                for key in ('train', 'validation'):
                    fold[key] = ['DFL-MAT-J03WMX' if m == MATCHES[0] else m for m in fold[key]]
        else:
            folds[0]['train'][0] = folds[0]['train'][1]
        with pytest.raises(ValueError):
            runner.validate_folds(folds)


def test_aggregate_keeps_six_matches_and_averages_paired_mask_batches():
    raw = _raw_scores(.4, .8)
    result = runner.aggregate_scores(raw)
    assert result['objective'] == pytest.approx(.6)
    assert result['missing'] == pytest.approx(.6)
    assert len(result['by_match']) == 6
    assert result['by_batch'] == {'batch_1': .4, 'batch_2': .8}
    assert result['means']['clean'] == pytest.approx(.6)
    del raw['batch_2']['by_match'][MATCHES[0]]
    with pytest.raises(ValueError, match='six whole matches'):
        runner.aggregate_scores(raw)
    with pytest.raises(ValueError, match='both fixed mask batches'):
        runner.aggregate_scores({'batch_1': raw['batch_1']})


def test_gate_requires_improvement_over_both_control_and_frozen_v6():
    v6 = runner.aggregate_scores(_raw_scores(.5, .5))
    control = runner.aggregate_scores(_raw_scores(.55, .55))
    candidate = runner.aggregate_scores(_raw_scores(.56, .56))
    gate = runner.comparison_gate(v6, control, candidate)
    assert gate['confirmation_eligible']
    assert not gate['automatic_promotion']
    assert not runner.comparison_gate(v6, candidate, control)['confirmation_eligible']
    assert not runner.comparison_gate(candidate, v6, control)['confirmation_eligible']


def test_gate_protects_scenarios_and_requires_each_batch_and_four_matches():
    reference = runner.aggregate_scores(_raw_scores())
    good = runner.aggregate_scores(_raw_scores(.52, .52))
    for failure, expected in (
        ('scenario', 'protect_scenarios_over_v6'),
        ('batch', 'batches_over_v6'),
        ('breadth', 'breadth_over_v6'),
        ('missing', 'missing_over_v6'),
        ('small_gain', 'gain_over_v6'),
    ):
        candidate = copy.deepcopy(good)
        if failure == 'scenario':
            candidate['means']['long'] = .489
        elif failure == 'batch':
            candidate['by_batch']['batch_2'] = .5
        elif failure == 'breadth':
            for m in MATCHES[3:]:
                candidate['by_match'][m] = {s: .5 for s in runner.WEIGHTS}
        elif failure == 'missing':
            candidate['missing'] = reference['missing']
        else:
            candidate['objective'] = reference['objective'] + .00199
        gate = runner.comparison_gate(reference, reference, candidate)
        assert not gate['checks'][expected]
        assert not gate['confirmation_eligible']


class _Budget:
    def check(self):
        pass

    def status(self, *args, **kwargs):
        pass


@pytest.mark.parametrize('variant', runner.VARIANTS)
def test_training_variants_share_authenticated_raw_start_and_1200_update_budget(
        tmp_path, monkeypatch, variant):
    from experiments import refinement_training

    parent = {'fold': 0, 'seed': 8101, 'name': 'member', 'spec': {'same': 'spec'},
              'train_matches': _folds()[0]['train'],
              'validation_matches': _folds()[0]['validation'],
              'parent_identity': {'seed': 12345}, 'state_path': 'raw_state_007200.pt',
              'state_sha256': 'rawhash', 'teacher_path': 'teacher.pt',
              'teacher_sha256': 'teacherhash'}
    protocol = {'data_dir': 'synthetic', 'settings': runner.SETTINGS}
    events = []

    class Session:
        steps = 0

        def advance_to(self, step):
            assert step - self.steps == 200
            events.append(('advance', step))
            self.steps = step

        def _progress(self):
            return {'steps': self.steps}

    session = Session()

    def build(data_dir, spec, train_matches, seed, cache):
        assert train_matches == parent['train_matches']
        assert not set(train_matches) & set(parent['validation_matches'])
        assert seed == parent['parent_identity']['seed']
        events.append(('build',))
        return session

    def load_raw(path, received, identity):
        assert received is session and path == parent['state_path']
        assert identity is parent['parent_identity']
        events.append(('raw_restore',))
        session.steps = 7200

    def configure(received, **kwargs):
        assert received is session and session.steps == 7200
        assert kwargs['variant'] == variant
        assert {k: kwargs[k] for k in runner.SETTINGS} == runner.SETTINGS
        assert (kwargs['teacher'] is not None) == (variant == 'masked_teacher')
        events.append(('configure',))
        return session

    def export(path, received, identity, use_tail_ema):
        assert received.steps == 8400 and use_tail_ema
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'synthetic exported model')
        events.append(('export',))

    monkeypatch.setattr(runner.campaign_training, 'build_session', build)
    monkeypatch.setattr(runner.campaign_training, 'load_training_state', load_raw)
    monkeypatch.setattr(refinement_training, 'configure_refinement', configure)
    monkeypatch.setattr(refinement_training, 'save_refinement_state', lambda *args: None)
    monkeypatch.setattr(refinement_training, 'export_refinement_checkpoint', export)
    monkeypatch.setattr(runner, '_check_hash', lambda *args: None)
    monkeypatch.setattr(runner, 'freeze_protocol', lambda *args: None)
    monkeypatch.setattr(runner, 'load_model_checkpoint', lambda path: (
        object(), {'training_config': {'train_matches': parent['train_matches'], 'steps': 8400}}))
    fit, path = runner._train_one(parent, variant, protocol, tmp_path, object(), _Budget())
    assert events[:3] == [('build',), ('raw_restore',), ('configure',)]
    assert [v[1] for v in events if v[0] == 'advance'] == [7400, 7600, 7800, 8000, 8200, 8400]
    assert fit['steps'] == 8400 and path.exists()
    assert fit['identity']['parent_state_sha256'] == 'rawhash'


def test_prediction_uses_same_frozen_masks_and_excluded_whole_matches(monkeypatch, tmp_path):
    parent = {'train_matches': list(MATCHES[2:]), 'validation_matches': list(MATCHES[:2]),
              'fold': 0, 'seed': 8101, 'name': 'member'}
    inputs = {batch: [{'match': m, 'period': period, 'scenario': scenario,
                       'input_sha256': f'{batch}-{m}-{period}-{scenario}'}
                      for m in MATCHES for period in (1, 2) for scenario in runner.WEIGHTS]
              for batch in runner.BATCHES}
    signatures = {batch: f'{batch}-manifest' for batch in runner.BATCHES}
    calls = []

    def predict(model, selected, output_dir, identity, progress_callback):
        batch = identity['batch']
        assert len(selected) == 16
        assert selected == [r for r in inputs[batch] if r['match'] in MATCHES[:2]]
        assert identity['input_manifest_sha256'] == signatures[batch]
        calls.append(batch)
        return selected

    monkeypatch.setattr(runner, 'load_model_checkpoint', lambda path: (object(), {}))
    monkeypatch.setattr(runner.campaign_evaluation, 'predict_records', predict)
    records = runner._predict_one(parent, 'control', {'input_signatures': signatures}, tmp_path,
                                  'synthetic.pt', {'identity': {}, 'checkpoint_sha256': 'model'},
                                  inputs, _Budget())
    assert calls == list(runner.BATCHES) and len(records) == 32
    parent['train_matches'] += [MATCHES[0]]
    with pytest.raises(ValueError, match='training ancestry'):
        runner._predict_one(parent, 'control', {'input_signatures': signatures}, tmp_path,
                            'synthetic.pt', {'identity': {}, 'checkpoint_sha256': 'model'},
                            inputs, _Budget())


def _reference_grid(tmp_path):
    n = 8
    arrays = {
        'frame_id': np.arange(n), 'gt': np.array([0, 0, 1, 1, 0, 0, 2, 2], dtype=np.int8),
        'neural_live': np.full(n, .99), 'neural_home': np.full(n, .99),
        'neural_boundary': np.zeros((n, 3, 3)), 'ball_valid': np.ones(n, dtype=bool),
        'nearest_ball_seconds': np.zeros(n), 'coarse_live': np.full(n, .1),
        'coarse_home': np.full(n, .1), 'ref_nearest_m': np.ones((n, 2)),
        'ref_second_m': np.ones((n, 2)), 'ref_motion_factor': np.ones((n, 2)),
        'ref_owner_observed': np.ones(n, dtype=bool), 'ref_activity': np.ones(n),
        'ref_activity_observed': np.ones(n, dtype=bool), 'ref_ball_motion': np.ones(n, dtype=bool),
        'ref_ball_speed': np.ones(n), 'ref_outside': np.zeros(n, dtype=bool),
        'ref_fps': np.asarray(25.), 'ref_pitch': np.array([105., 68.]),
    }
    cfg = baseline_decoder({'ensemble': {'neural_weight': 1.}, 'calibration': {},
                            'decoder': {'switch_cost': 0., 'boundary_scale': 0.}})
    arrays['v6_labels'] = decode_arrays(arrays, cfg)
    source = tmp_path / 'rich.npz'
    np.savez(source, **arrays)
    prediction = tmp_path / 'prediction.npz'
    np.savez(prediction, **{k: v for k, v in arrays.items() if k.startswith('neural_')})
    protocol = {'selection': {'members': [{'id': 'mean'}, {'id': 'attention'}],
                              'blend_weight': .5, 'decoder': cfg}}
    rich, predictions = [], []
    for batch in runner.BATCHES:
        for match in MATCHES:
            for period in (1, 2):
                for scenario in runner.WEIGHTS:
                    row = {'batch': batch, 'scope': 'inner', 'match': match, 'period': period,
                           'scenario': scenario, 'path': str(source), 'sha256': sha256_file(source),
                           'source_input_sha256': f'{batch}-{match}-{period}-{scenario}',
                           'n_frames': n}
                    rich.append(row)
                    for variant in runner.VARIANTS:
                        for seed in runner.SEEDS:
                            for member in ('mean', 'attention'):
                                predictions.append({**row, 'variant': variant, 'seed': seed,
                                                    'member': member,
                                                    'input_sha256': row['source_input_sha256'],
                                                    'prediction_path': str(prediction),
                                                    'prediction_sha256': sha256_file(prediction)})
    return protocol, rich, predictions, arrays


def test_complete_synthetic_reference_grid_keeps_frozen_decoder_and_coarse_context(
        tmp_path, monkeypatch, capsys):
    protocol, rich, predictions, arrays = _reference_grid(tmp_path)
    output = tmp_path / 'output'
    result = runner.evaluate_predictions(protocol, rich, predictions, output)
    assert result['decision'] == 'retain_v6'
    assert not result['challenge_submission_modified']
    assert not result['confirmation_candidates']
    manifest = read_record(output / 'reference/manifest.json')
    assert len(manifest['records']) == 96
    with np.load(output / 'reference/labels.npz') as saved:
        for row in manifest['records']:
            for variant in ('v6', *runner.VARIANTS):
                labels = saved[row['prediction_keys'][variant]]
                np.testing.assert_array_equal(labels, arrays['v6_labels'])
                assert not np.array_equal(labels, arrays['gt'])
    for variant in runner.VARIANTS:
        assert result['summary'][variant]['objective'] == pytest.approx(
            result['summary']['v6']['objective'])
    # An identical rerun must reuse the exact portable result.
    assert runner.evaluate_predictions(protocol, rich, predictions, output) == result
    # Label-only API and CLI reproduce every metric without invoking training.
    assert runner.evaluate_reference(output / 'reference') == result
    monkeypatch.setattr(runner, 'run', lambda *args, **kwargs: pytest.fail('training invoked'))
    monkeypatch.setattr(sys, 'argv', ['possession_refinement', '--score-reference',
                                     str(output / 'reference')])
    runner.main()
    assert json.loads(capsys.readouterr().out) == result
    diagnostics = result['spell_diagnostics']['v6']['batch_1']['clean']
    assert diagnostics['counts'] == {'gt_spells': 36, 'pred_spells': 12,
                                     'matched_at_0_5': 0, 'missed_at_0_5': 36,
                                     'spurious_at_0_5': 12}
    assert diagnostics['duration_bins']['under_1s']['gt'] == 36
    assert diagnostics['duration_bins']['under_1s']['recall_at_0_5'] == 0.
    with np.load(output / 'reference/labels.npz') as saved:
        decoded_labels = {key: saved[key] for key in saved.files}
    assert runner.score_reference(decoded_labels, manifest) == result
    # Native halves remain separate even when both end/start with the same state.
    assert diagnostics['counts']['pred_spells'] == 2 * len(MATCHES)
    with (output / 'reference/labels.npz').open('ab') as handle:
        handle.write(b'corruption')
    with pytest.raises(ValueError, match='hash|checksum'):
        runner.evaluate_reference(output / 'reference')


def test_prediction_grid_rejects_missing_or_duplicate_member_seed_outputs(tmp_path):
    protocol, rich, predictions, _ = _reference_grid(tmp_path)
    with pytest.raises(ValueError, match='incomplete refinement prediction grid'):
        runner.evaluate_predictions(protocol, rich, predictions[:-1], tmp_path / 'missing')
    with pytest.raises(ValueError, match='duplicate refinement prediction'):
        runner.evaluate_predictions(protocol, rich, predictions + [predictions[0]], tmp_path / 'dup')


def test_all_members_using_wrong_mask_are_rejected_even_when_internally_consistent(tmp_path):
    protocol, rich, predictions, _ = _reference_grid(tmp_path)
    for row in predictions:
        row['input_sha256'] = 'same-wrong-cache-for-all-members'
    with pytest.raises(ValueError, match='input mask differs'):
        runner.evaluate_predictions(protocol, rich, predictions, tmp_path / 'output')


def test_prediction_grid_requires_exact_variant_seed_member_keys(tmp_path):
    protocol, rich, predictions, _ = _reference_grid(tmp_path)
    predictions[-1] = {**predictions[-1], 'seed': 9999}
    with pytest.raises(ValueError, match='incomplete refinement prediction grid'):
        runner.evaluate_predictions(protocol, rich, predictions, tmp_path / 'output')


def test_changing_gt_changes_scores_but_cannot_change_predictions(tmp_path):
    protocol, rich, predictions, arrays = _reference_grid(tmp_path)
    first = runner.evaluate_predictions(protocol, rich, predictions, tmp_path / 'first')
    changed = dict(arrays)
    changed['gt'] = np.zeros(len(arrays['gt']), dtype=np.int8)
    source = tmp_path / 'changed_gold.npz'
    np.savez(source, **changed)
    altered_rich = [{**row, 'path': str(source), 'sha256': sha256_file(source)} for row in rich]
    second = runner.evaluate_predictions(protocol, altered_rich, predictions, tmp_path / 'second')
    assert first['summary']['control']['objective'] != second['summary']['control']['objective']
    with (np.load(tmp_path / 'first/reference/labels.npz') as a,
          np.load(tmp_path / 'second/reference/labels.npz') as b):
        for key in a.files:
            if not key.startswith('gt_'):
                np.testing.assert_array_equal(a[key], b[key])


def test_active_coarse_context_and_decoder_are_frozen_for_every_variant(tmp_path):
    from experiments.coarse_context import blend_coarse_context

    protocol, rich, predictions, arrays = _reference_grid(tmp_path)
    n = len(arrays['gt'])
    fine = {'neural_live': np.full(n, .8), 'neural_home': np.full(n, .8),
            'neural_boundary': np.zeros((n, 3, 3))}
    arrays.update(fine)
    arrays['ball_valid'] = np.array([True, True] + [False] * (n - 2))
    arrays['nearest_ball_seconds'] = np.where(arrays['ball_valid'], 0., 10.)
    arrays['coarse_live'] = np.full(n, .001)
    arrays['coarse_home'] = np.full(n, .001)
    mixed = blend_coarse_context(arrays, arrays['coarse_live'], arrays['coarse_home'])
    arrays.update({k: mixed[k] for k in fine})
    expected = decode_arrays(arrays, protocol['selection']['decoder'])
    arrays['v6_labels'] = expected
    np.testing.assert_array_equal(expected, [0, 0, 2, 2, 2, 2, 2, 2])
    source = tmp_path / 'active_coarse.npz'
    np.savez(source, **arrays)
    model_prediction = tmp_path / 'active_fine_prediction.npz'
    np.savez(model_prediction, **fine)
    rich = [{**r, 'path': str(source), 'sha256': sha256_file(source)} for r in rich]
    predictions = [{**r, 'prediction_path': str(model_prediction),
                    'prediction_sha256': sha256_file(model_prediction)} for r in predictions]
    output = tmp_path / 'output'
    runner.evaluate_predictions(protocol, rich, predictions, output)
    manifest = read_record(output / 'reference/manifest.json')
    with np.load(output / 'reference/labels.npz') as saved:
        for row in manifest['records']:
            for variant in runner.VARIANTS:
                np.testing.assert_array_equal(saved[row['prediction_keys'][variant]], expected)

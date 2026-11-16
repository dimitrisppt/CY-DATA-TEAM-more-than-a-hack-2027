"""Write a readable refinement comparison, optionally when a running study ends."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from experiments.campaign_protocol import read_record
from experiments.possession_refinement import evaluate_reference
from possession.storage import write_json


def _short_recall(result, variant):
    # Report one fixed clean batch. The two clean copies are identical inputs.
    bins = result['spell_diagnostics'][variant]['batch_1']['clean']['duration_bins']
    gt = sum(bins[key]['gt'] for key in ('under_1s', '1_to_2s'))
    matched = sum(bins[key]['matched_at_0_5'] for key in ('under_1s', '1_to_2s'))
    return matched, gt, matched / gt if gt else 0.


def render(result):
    """Render measured outcomes without treating an eligible screen as a release."""
    rows = result['summary']
    lines = [
        '# Possession-refinement comparison', '',
        'The delivered model and submission remain v6. These are public-data development scores.',
        '',
        '| Variant | Weighted spell F1 | Delta vs v6 | Delta vs control | Missing-ball score |',
        '|---|---:|---:|---:|---:|',
    ]
    for name in ('v6', 'control', 'spell_balanced', 'masked_teacher'):
        row = rows[name]
        lines.append(f"| {name} | {row['objective']:.6f} | "
                     f"{row['objective'] - rows['v6']['objective']:+.6f} | "
                     f"{row['objective'] - rows['control']['objective']:+.6f} | "
                     f"{row['missing']:.6f} |")
    lines.extend(['', '## Scenario scores', '',
                  '| Variant | Clean | Short gaps | Long gaps | Replayed gaps |',
                  '|---|---:|---:|---:|---:|'])
    for name in ('v6', 'control', 'spell_balanced', 'masked_teacher'):
        values = rows[name]['means']
        lines.append(f"| {name} | " + ' | '.join(f'{values[s]:.6f}' for s in
                     ('clean', 'short', 'long', 'replay')) + ' |')
    lines.extend(['', '## Short possessions and fragmentation', '',
                  'Clean batch 1, pooled event counts at temporal IoU 0.5.', '',
                  '| Variant | Under-2s matched / true | Under-2s recall | Spurious predictions |',
                  '|---|---:|---:|---:|'])
    for name in ('v6', 'control', 'spell_balanced', 'masked_teacher'):
        matched, gt, recall = _short_recall(result, name)
        count = result['spell_diagnostics'][name]['batch_1']['clean']['counts']
        lines.append(f"| {name} | {matched} / {gt} | {recall:.1%} | "
                     f"{count['spurious_at_0_5']} |")
    lines.extend(['', '## Fixed advancement checks', ''])
    for name, gate in result['gates'].items():
        failed = [key for key, value in gate['checks'].items() if not value]
        if gate['confirmation_eligible']:
            lines.append(f'- `{name}` passed the development screen. Further confirmation '
                         'and a separate guard are required before considering a release.')
        else:
            lines.append(f'- `{name}` did not pass. Failed checks: ' + ', '.join(failed) + '.')
    lines.extend(['', '## Interpretation', '', result['historical_reuse_caveat'], '',
                  ('The control isolates extra training and its final EMA. Improvements over v6 '
                  'alone do not demonstrate a benefit from the auxiliary technique. No '
                  'hyperparameter or decoder changes were made after observing these scores.'), '',
                  f"Protocol SHA256: `{result['protocol_sha256']}`", ''])
    return '\n'.join(lines)


def report(directory, output):
    directory, output = Path(directory), Path(output)
    recorded = read_record(directory / 'result.json')
    if recorded is None:
        raise FileNotFoundError('complete study result is missing')
    reproduced = evaluate_reference(directory / 'reference')
    if recorded != reproduced:
        raise ValueError('label-only reproduction differs from the recorded study result')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.name}.tmp')
    temporary.write_text(render(reproduced), encoding='utf-8')
    os.replace(temporary, output)
    write_json(output.with_suffix('.verification.json'), {
        'status': 'complete', 'label_only_reproduction_equal': True,
        'protocol_sha256': recorded['protocol_sha256'],
        'decision': recorded['decision'],
        'confirmation_candidates': recorded['confirmation_candidates'],
        'report': str(output.resolve()),
    })
    return output


def wait_and_report(directory, output):
    """Observe one already running job, without restarting or changing its budget."""
    directory = Path(directory)
    while True:
        status = json.loads((directory / 'status.json').read_text())
        if status['status'] == 'complete':
            return report(directory, output)
        if status['status'] != 'running':
            raise RuntimeError(f"study stopped with status {status['status']}")
        try:
            os.kill(status['pid'], 0)
        except ProcessLookupError as exc:
            raise RuntimeError('study process ended before completion, saved work is resumable') from exc
        if time.time() > status.get('deadline_unix', 0) + 300:
            raise RuntimeError('study exceeded its recorded wall-clock deadline')
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path,
                        default=Path('outputs/possession-refinement/comparison.md'))
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    action = wait_and_report if args.wait else report
    print(action(args.study_dir, args.output), flush=True)


if __name__ == '__main__':
    main()

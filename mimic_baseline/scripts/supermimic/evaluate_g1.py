"""Evaluate a ManagerBased G1 checkpoint on exactly all 40 LAFAN1 clips."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import subprocess
import sys


def validate_results(payload, expected_names):
    rows = payload['clips']
    if len(rows) != 40 or {r['name'] for r in rows} != set(expected_names):
        raise ValueError('Evaluation must contain exactly all 40 source clips')
    for row in rows:
        if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in row.values()):
            raise ValueError(f"Nonfinite evaluation metrics: {row['name']}")
    return {'clips': len(rows), 'completed': sum(bool(r['completed']) for r in rows),
            'completion_rate': sum(bool(r['completed']) for r in rows) / len(rows),
            'mean_normalized_return': sum(r['normalized_return'] for r in rows) / len(rows),
            'failed_clips': [r['name'] for r in rows if not r['completed']]}


def evaluate(checkpoint, data_root, output, *, dry_run=False):
    import torch

    checkpoint, data_root, output = (Path(p).resolve() for p in (checkpoint, data_root, output))
    if output.exists():
        raise FileExistsError(f'Refusing to reuse evaluation output: {output}')
    names = sorted(p.stem for p in (data_root / 'lafan1_g1').glob('*.csv'))
    if len(names) != 40:
        raise ValueError('Expected exactly 40 LAFAN1 source clips')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved.get('format') != 'supermimic-g1-v1' or saved['config'].get('robot_name') != 'g1':
        raise ValueError('Expected an original ManagerBased G1 checkpoint')
    if not saved.get('policy') or not all(torch.isfinite(v).all() for v in saved['policy'].values()):
        raise ValueError('Missing or nonfinite policy parameters')
    metadata = {'checkpoint': checkpoint.name, 'checkpoint_iteration': saved['iteration'],
                'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'backend': saved.get('provenance', {}).get('backend', 'ovphysx'),
                'expected_clips': names, 'protocol': 'original deterministic tracking failures; original unpadded clip lengths',
                'recovery_enabled': saved['config'].get('recovery_enabled', False),
                'action_delay': {'actuator_type': saved['config'].get('actuator_type'),
                                 'max_delay': saved['config'].get('max_delay')},
                'seed': saved['config'].get('seed', 0)}
    del saved
    command = [sys.executable, '-u', str(Path(__file__).with_name('train_g1.py')),
               '--mode', 'evaluate', '--resume', str(checkpoint), '--data-root', str(data_root),
               '--log-dir', str(output)]
    if dry_run:
        return {'preflight_passed': True, 'command': command, **metadata}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'evaluation_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    try:
        with (output / 'evaluation.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = validate_results(json.loads((output / 'perclip.json').read_text(encoding='utf-8')), names)
        summary.update(metadata)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        return summary
    except BaseException as error:
        (output / 'failure.json').write_text(json.dumps({'error': str(error)}, indent=2), encoding='utf-8')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='Validate checkpoint and data without launching the simulator')
    print(json.dumps(evaluate(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()

"""Run a pinned Newton recovery experiment, then evaluate all 40 LAFAN1 clips."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def run_experiment(data_root, run_dir, num_envs, iterations, source_revision):
    import torch

    data_root, run_dir = Path(data_root).resolve(), Path(run_dir).resolve()
    names = {p.stem for p in (data_root / 'lafan1_g1').glob('*.csv')}
    if len(names) != 40:
        raise ValueError('Expected exactly 40 LAFAN1 source clips')
    if num_envs < 1 or iterations < 1:
        raise ValueError('num_envs and iterations must be positive')
    if len(source_revision) != 40 or any(c not in '0123456789abcdef' for c in source_revision):
        raise ValueError('source_revision must be a full Git SHA')
    run_dir.mkdir(parents=True, exist_ok=False)
    entry = Path(__file__).with_name('train_g1.py')
    checkpoint = run_dir / 'train' / f'model_{iterations}.pt'
    common = [sys.executable, '-u', str(entry), '--data-root', str(data_root)]
    train = common + ['--mode', 'train', '--backend', 'newton', '--recovery',
                      '--actuator-type', 'delayed_pd', '--max-delay', '4',
                      '--num-envs', str(num_envs), '--iterations', str(iterations),
                      '--save-every', '100', '--log-every', '10', '--log-dir', str(run_dir / 'train')]
    evaluate = common + ['--mode', 'evaluate', '--resume', str(checkpoint),
                         '--log-dir', str(run_dir / 'evaluate')]
    write_json(run_dir / 'experiment.json', dict(source_revision=source_revision,
               num_envs=num_envs, iterations=iterations, train_command=train,
               evaluate_command=evaluate, expected_clips=sorted(names)))
    phase = 'training'
    status = {'phase': phase, 'target_iteration': iterations, 'source_revision': source_revision}
    try:
        for command, phase in [(train, 'training'), (evaluate, 'evaluating')]:
            status.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(run_dir / 'status.json', status)
            with (run_dir / f'{phase}.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            if phase == 'training':
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                if saved['iteration'] != iterations or saved['format'] != 'supermimic-g1-v1':
                    raise ValueError('Final checkpoint iteration/format mismatch')
                cfg = saved['config']
                if not (cfg['recovery_enabled'] and cfg['self_collisions'] and
                        cfg['actuator_type'] == 'delayed_pd' and cfg['max_delay'] == 4 and
                        cfg['num_envs'] == num_envs and saved['provenance']['backend'] == 'newton'):
                    raise ValueError('Final checkpoint does not match the requested recovery experiment')
                if not all(torch.isfinite(v).all() for key in ('wm', 'policy', 'critic') for v in saved[key].values()):
                    raise ValueError('Nonfinite final model parameters')
                del saved
                with checkpoint.open('rb') as handle:
                    status['checkpoint_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
                status['checkpoint_iteration'] = iterations
        rows = json.loads((run_dir / 'evaluate/perclip.json').read_text(encoding='utf-8'))['clips']
        if len(rows) != 40 or {r['name'] for r in rows} != names:
            raise ValueError('Evaluation did not produce exactly the 40 source clips')
        status.update(phase='complete', evaluated_clips=40, completed_clips=sum(bool(r['completed']) for r in rows),
                      updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(run_dir / 'status.json', status)
        return status
    except BaseException as error:
        status.update(phase=f'{phase}_failed', error=str(error), updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(run_dir / 'status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, required=True)
    parser.add_argument('--iterations', type=int, default=3000)
    parser.add_argument('--source-revision', required=True)
    args = parser.parse_args()
    print(json.dumps(run_experiment(**vars(args))))

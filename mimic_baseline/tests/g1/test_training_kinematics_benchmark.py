"""Benchmark must run backward and report all output/input numerical comparisons."""
import pytest

from test_kinematics import make_urdf


def test_benchmark_runs_real_backward_and_reports_every_tensor(tmp_path):
    from mimic_baseline.tasks.supermimic.validation import training_kinematics

    report = training_kinematics.benchmark(
        make_urdf(tmp_path), device='cpu', batch=7, chunks=(3, 5), repeats=1,
        dtype='float64',
    )
    assert report['passed']
    assert report['batch'] == 7
    assert [v['name'] for v in report['variants']] == ['reference', 'batched', 'chunk_3', 'chunk_5']
    for variant in report['variants']:
        assert variant['status'] == 'complete'
        sample = variant['runs'][0]
        assert sample['forward_wall_seconds'] > 0
        assert sample['backward_wall_seconds'] > 0
        assert sample['forward_cuda_seconds'] is None
        assert sample['peak_allocated_bytes'] is None
        assert sample['rng_unchanged']
        assert len(sample['numerics']['outputs']) == 4
        assert len(sample['numerics']['gradients']) == 6
        assert all(tensor['reference_max_abs'] > 0 for tensor in sample['numerics']['gradients'])
        assert sample['numerics']['passed']
    # Runtime/evidence are public; no input/output filesystem locations escape.
    assert str(tmp_path) not in str(report)


@pytest.mark.parametrize('kwargs', [
    {'batch': 0}, {'batch': 1048577}, {'chunks': ()}, {'chunks': (0,)},
    {'chunks': (3, 3)}, {'repeats': 0}, {'repeats': 4}, {'dtype': 'float16'},
    {'device': 'mps'},
])
def test_benchmark_rejects_unbounded_or_unsupported_arguments(tmp_path, kwargs):
    from mimic_baseline.tasks.supermimic.validation import training_kinematics

    with pytest.raises(ValueError):
        training_kinematics.benchmark(make_urdf(tmp_path), **kwargs)

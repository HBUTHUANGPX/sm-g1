"""Isolated FK forward/backward benchmark; never starts or resumes training."""

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch

from ..core.rotations import quat_to_mat
from ..robots.g1.kinematics import G1BatchedKinematics, G1Kinematics, G1TrainingKinematics


_OUTPUT_NAMES = ('position', 'rotation', 'linear_velocity', 'angular_velocity')
_INPUT_NAMES = ('p', 'R', 'v', 'w', 'q', 'qd')
_DEFAULT_CHUNKS = (4096, 8192, 16384, 32768)


def _validate(device, batch, chunks, repeats, dtype, seed):
    if not isinstance(batch, int) or isinstance(batch, bool) or not 1 <= batch <= 1048576:
        raise ValueError('batch must be an integer between 1 and 1048576')
    if not isinstance(repeats, int) or isinstance(repeats, bool) or not 1 <= repeats <= 3:
        raise ValueError('repeats must be between 1 and 3')
    if not 1 <= len(chunks) <= 8 or len(set(chunks)) != len(chunks):
        raise ValueError('provide between 1 and 8 distinct chunk sizes')
    if any(not isinstance(c, int) or isinstance(c, bool) or not 1 <= c <= 1048576 for c in chunks):
        raise ValueError('chunk sizes must be positive integers no greater than 1048576')
    if dtype not in ('float32', 'float64'):
        raise ValueError('dtype must be float32 or float64')
    if not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError('seed must be an integer in [0, 2**63)')
    parsed = torch.device(device)
    if parsed.type not in ('cpu', 'cuda'):
        raise ValueError('benchmark supports CPU and CUDA only')
    return parsed


def _fixture(batch, dtype, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    inputs = [torch.randn(batch, *shape, generator=generator, dtype=dtype)
              for shape in ((3,), (3, 3), (3,), (3,), (29,), (29,))]
    quat = torch.nn.functional.normalize(torch.randn(batch, 4, generator=generator, dtype=dtype), dim=-1)
    inputs[1] = quat_to_mat(quat)
    inputs[4] *= 0.4
    return inputs


def _loss(outputs):
    # Each of the four returned fields contributes a deterministic nonzero
    # gradient; means keep loss magnitude independent of the batch size.
    return sum((index + 1) * tensor.square().mean() for index, tensor in enumerate(outputs))


def _measure(function, device):
    is_cuda = device.type == 'cuda'
    if is_cuda:
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    begin = time.perf_counter()
    if is_cuda:
        start.record(torch.cuda.current_stream(device))
    result = function()
    if is_cuda:
        end.record(torch.cuda.current_stream(device))
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - begin
    cuda = start.elapsed_time(end) / 1000 if is_cuda else None
    return result, wall, cuda


def _difference(actual, reference, name, *, atol, rtol):
    # CPU-only comparison: no reference graph or reference tensors on the GPU.
    actual = actual.detach().cpu()
    difference = (actual - reference).double()
    reference64 = reference.double()
    max_abs = difference.abs().max().item()
    reference_max = reference64.abs().max().item()
    relative_l2 = (torch.linalg.vector_norm(difference) /
                   torch.linalg.vector_norm(reference64).clamp_min(torch.finfo(torch.float64).tiny)).item()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    return {
        'name': name,
        'max_abs': max_abs if math.isfinite(max_abs) else None,
        'reference_max_abs': reference_max if math.isfinite(reference_max) else None,
        'max_abs_over_reference_max': (max_abs / max(reference_max, torch.finfo(torch.float64).tiny)) if finite else None,
        'relative_l2': relative_l2 if math.isfinite(relative_l2) else None,
        'finite': finite,
        'allclose': finite and torch.allclose(actual, reference, atol=atol, rtol=rtol),
    }


def _run(kin, fixture, device, reference, batch, dtype):
    inputs = [x.to(device=device, copy=True).requires_grad_() for x in fixture]
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    memory = {key: None for key in ('baseline_allocated_bytes', 'baseline_reserved_bytes',
                                  'peak_allocated_bytes', 'peak_reserved_bytes')}
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        memory['baseline_allocated_bytes'] = torch.cuda.memory_allocated(device)
        memory['baseline_reserved_bytes'] = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
    outputs, forward_wall, forward_cuda = _measure(lambda: kin.forward(*inputs), device)
    loss = _loss(outputs)
    _, backward_wall, backward_cuda = _measure(loss.backward, device)
    if device.type == 'cuda':
        memory['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        memory['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
    gradients = [x.grad for x in inputs]
    if any(x is None for x in gradients):
        raise AssertionError('FK did not propagate all six input gradients')
    rng_unchanged = torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        rng_unchanged = rng_unchanged and torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    if reference is None:
        reference = tuple(tuple(x.detach().to(device='cpu', copy=True) for x in group)
                          for group in (outputs, gradients))
    rtol, atol = (2e-4, 2e-5) if dtype == 'float32' else (1e-10, 1e-11)
    numerics = {
        'outputs': [_difference(x, ref, name, atol=atol, rtol=rtol)
                    for x, ref, name in zip(outputs, reference[0], _OUTPUT_NAMES)],
        'gradients': [_difference(x, ref, name, atol=atol / batch, rtol=rtol)
                      for x, ref, name in zip(gradients, reference[1], _INPUT_NAMES)],
        'output_atol': atol, 'gradient_atol': atol / batch, 'rtol': rtol,
    }
    numerics['passed'] = all(row['allclose'] for group in ('outputs', 'gradients')
                             for row in numerics[group])
    return {
        'forward_wall_seconds': forward_wall, 'backward_wall_seconds': backward_wall,
        'forward_cuda_seconds': forward_cuda, 'backward_cuda_seconds': backward_cuda,
        'loss': loss.item() if torch.isfinite(loss) else None, 'rng_unchanged': bool(rng_unchanged),
        **memory, 'numerics': numerics,
    }, reference


def _release(device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def benchmark(urdf, device='cuda:0', *, batch=131072, chunks=_DEFAULT_CHUNKS,
              repeats=1, seed=431, dtype='float32'):
    """Compare isolated full-batch FK and backward without simultaneous graphs.

    Peak CUDA memory includes inputs, outputs, loss, backward and FK buffers;
    baseline is measured after allocating inputs. CPU reference snapshots are
    ephemeral and never serialized. Loss construction is outside both timing
    intervals but inside the peak-memory interval. Each repetition starts with
    fresh input tensors and an empty CUDA cache; this is a diagnostic harness,
    not a modification of the training allocator or training configuration.
    """
    device = _validate(device, batch, chunks, repeats, dtype, seed)
    tensor_dtype = getattr(torch, dtype)
    fixture = _fixture(batch, tensor_dtype, seed)
    variants = [('reference', G1Kinematics, None), ('batched', G1BatchedKinematics, None)]
    variants += [(f'chunk_{chunk}', G1TrainingKinematics, chunk) for chunk in chunks]
    results, reference = [], None
    for name, cls, chunk in variants:
        _release(device)
        kin = cls(urdf, str(device), **({'chunk_size': chunk} if chunk else {}))
        for field in ('off_p', 'off_R', 'axes', 'q_lo', 'q_hi'):
            setattr(kin, field, getattr(kin, field).to(tensor_dtype))
        record = {'name': name, 'chunk_size': chunk, 'status': 'complete', 'runs': []}
        try:
            warmup = [x[:min(256, batch)].to(device=device, copy=True).requires_grad_() for x in fixture]
            _loss(kin.forward(*warmup)).backward()
            del warmup
            _release(device)
            for _ in range(repeats):
                result, reference = _run(kin, fixture, device, reference, batch, dtype)
                record['runs'].append(result)
                _release(device)
        except torch.cuda.OutOfMemoryError:
            record['status'] = 'out_of_memory'
        results.append(record)
        del kin
        _release(device)
        if name == 'reference' and reference is None:
            break  # No numerical reference exists; do not claim comparisons.
    return {
        'passed': len(results) == len(variants) and all(
            item['status'] == 'complete' and len(item['runs']) == repeats and all(
                run['numerics']['passed'] and run['rng_unchanged'] for run in item['runs'])
            for item in results),
        'batch': batch, 'chunks': list(chunks), 'repeats': repeats, 'seed': seed, 'dtype': dtype,
        'warmup_batch': min(256, batch), 'loss': 'sum((field_index+1)*field.square().mean())',
        'runtime': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'device_type': device.type,
                    'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                    'cpu_threads': torch.get_num_threads()},
        'variants': results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch', type=int, default=131072)
    parser.add_argument('--chunks', type=int, nargs='+', default=list(_DEFAULT_CHUNKS))
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, default=431)
    parser.add_argument('--dtype', choices=('float32', 'float64'), default='float32')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        report = benchmark(args.urdf, args.device, batch=args.batch, chunks=tuple(args.chunks),
                           repeats=args.repeats, seed=args.seed, dtype=args.dtype)
    except ValueError as error:
        parser.error(str(error))
    serialized = json.dumps(report, separators=(',', ':'), allow_nan=False)
    if args.output:
        args.output.write_text(serialized + '\n', encoding='utf-8')
    print(serialized)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Decoder-only physical-gradient caps for clean recurrent WM updates."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from mimic_baseline.tasks.supermimic.learning.distributed import DistributedContext


@torch.no_grad()
def cap_decoder_physics_gradients(
    decoders: dict[str, nn.Module],
    reconstruction_gradients: tuple[torch.Tensor | None, ...],
    ratio: float,
    *,
    distributed: DistributedContext | None = None,
) -> dict[str, torch.Tensor]:
    """Cap each decoder's entire parameter vector before the common norm clip.

    ``.grad`` must already contain the rank-averaged total gradient. Average
    the weighted reconstruction contribution in one additional collective, then
    obtain the physical contribution by subtraction. Directly retaining the
    smaller reconstruction gradient avoids total-minus-physics cancellation.
    Shared-module and input gradients are unchanged.
    """
    groups = [(name, tuple(module.parameters())) for name, module in decoders.items()]
    parameters = tuple(p for _, group in groups for p in group)
    reconstruction = torch.cat(
        tuple(
            (torch.zeros_like(p) if g is None else g).reshape(-1)
            for p, g in zip(parameters, reconstruction_gradients, strict=True)
        )
    )
    # Include unused parameters as zeros so all ranks have the same layout,
    # including explicitly disabled reconstruction terms.
    if distributed is not None and distributed.enabled:
        dist.all_reduce(reconstruction, op=dist.ReduceOp.SUM)
        reconstruction.div_(distributed.world_size)

    metrics = {}
    offset = 0
    for name, group in groups:
        count = sum(p.numel() for p in group)
        gr = reconstruction[offset : offset + count]
        total = torch.cat(
            tuple(
                (torch.zeros_like(p) if p.grad is None else p.grad).reshape(-1)
                for p in group
            )
        )
        gp = total - gr
        pnorm = torch.linalg.vector_norm(gp)
        rnorm = torch.linalg.vector_norm(gr)
        scale = (ratio * rnorm / (pnorm + 1e-12)).clamp(max=1.0)
        capped = gr + scale * gp
        cursor = 0
        for parameter in group:
            size = parameter.numel()
            if parameter.grad is not None:
                parameter.grad.copy_(capped[cursor : cursor + size].view_as(parameter))
            cursor += size
        offset += count
        metrics.update(
            {
                f"Gradient/{name}_decoder_physics_norm": pnorm,
                f"Gradient/{name}_decoder_reconstruction_norm": rnorm,
                f"Gradient/{name}_decoder_physics_scale": scale,
                f"Gradient/{name}_decoder_capped_physics_norm": scale * pnorm,
            }
        )
    return metrics

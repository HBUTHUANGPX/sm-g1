"""Fixed-size actuator delay storage without per-step host/device synchronization."""

from collections.abc import Sequence

import torch


class ActuatorDelayBuffer:
    """The compute/reset subset of DelayBuffer used by delayed actuators.

    Delay bounds are supplied by the actuator configuration. History is read
    through a ring index rather than reordered on every physics step. A reset
    invalidates old entries by length, so no clearing or first-frame broadcast
    is needed: an unavailable delay selects the oldest post-reset input.
    """

    def __init__(self, history_length: int, batch_size: int, device: str, *, delay_step_size: int = 1):
        # Native delayed actuators sample ticks; expand them to physics steps here.
        self._delay_step_size = delay_step_size
        self._capacity = history_length * delay_step_size + 1
        self._batch_ids = torch.arange(batch_size, device=device)
        self._lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._time_lags = torch.zeros_like(self._lengths)
        self._head = -1
        self._buffer = None
        self._read_indices = None

    @property
    def time_lags(self) -> torch.Tensor:
        """Per-environment lag, also used by the paired ERP transport."""
        return self._time_lags

    def set_time_lag(self, time_lag: int | torch.Tensor, batch_ids: Sequence[int] | slice | None = None):
        if isinstance(time_lag, torch.Tensor):
            time_lag = time_lag.to(self._time_lags)
        self._time_lags[slice(None) if batch_ids is None else batch_ids] = time_lag * self._delay_step_size

    def reset(self, batch_ids: Sequence[int] | slice | None = None):
        self._lengths[slice(None) if batch_ids is None else batch_ids] = 0

    def compute(self, data: torch.Tensor, *, read_indices: torch.Tensor | None = None) -> torch.Tensor:
        """Append and read, optionally reusing indices from a paired buffer.

        Paired buffers must share capacity, reset rows, delays and update count.
        The delayed PD actuator already maintains that invariant for its three
        setpoints. Independent callers keep the original compute behavior.
        """
        if self._buffer is None:
            self._buffer = data.new_empty((self._capacity, *data.shape))
        self._head = (self._head + 1) % self._capacity
        self._buffer[self._head].copy_(data)
        self._lengths.add_(1).clamp_(max=self._capacity)
        if read_indices is None:
            lag = torch.minimum(self._time_lags, self._lengths - 1)
            read_indices = (self._head - lag).remainder(self._capacity)
        self._read_indices = read_indices
        # Advanced indexing already returns independent storage; no clone needed.
        return self._buffer[read_indices, self._batch_ids]

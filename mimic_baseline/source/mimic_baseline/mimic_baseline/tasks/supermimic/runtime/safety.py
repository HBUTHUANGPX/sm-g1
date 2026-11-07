"""Project-side PhysX failure propagation and periodic CUDA cache release."""

from __future__ import annotations

import torch


class PhysicsFatalError(RuntimeError):
    """The native scene cannot produce valid transitions anymore."""


class PhysxHealthGuard:
    """Latch native failures; Omniverse swallows exceptions raised in callbacks.

    Type 1 includes allocator OOM, not only ordinary PhysX validation errors.
    Bridge the latched exception into the existing manager callback mechanism
    so a failed physics substep stops before environment reset/reward work.
    """

    _FATAL_MESSAGES = (
        "failed to allocate", "out of memory", "aborting simulation",
        "scene state is corrupted", "simulation cannot continue",
        "unable to create scene", "no physics scene created",
        "cuda error", "illegal memory access",
    )

    def __init__(self, stream, store_exception):
        self._error = None
        self._store_exception = store_exception
        self._subscription = stream.create_subscription_to_push(
            self._on_error, name="supermimic_native_physics_health",
        )

    @classmethod
    def subscribe(cls):
        # Simulator imports must remain after SimulationApp startup.
        import omni.physx
        from isaaclab_physx.physics.physx_manager import PhysxManager

        return cls(omni.physx.get_physx_interface().get_error_event_stream(),
                   PhysxManager.store_callback_exception)

    def _on_error(self, event):
        message = str(event.payload.get("errorString", ""))
        if self._error is None and (
            int(event.type) == 2 or any(token in message.lower() for token in self._FATAL_MESSAGES)
        ):
            self._error = PhysicsFatalError(f"Native PhysX fatal (event={int(event.type)}): {message}")
            self._store_exception(self._error)

    def check(self):
        if self._error is not None:
            raise self._error

    def close(self):
        if self._subscription is not None:
            self._subscription.unsubscribe()
            self._subscription = None


class PeriodicCudaCache:
    """Release idle Torch cache on a fixed outer-training-iteration clock.

    The first collection clears once, including after resume. No device-memory
    query or allocator limit is needed; native PhysX fatal handling stays active
    independently because a periodic schedule cannot guarantee free headroom.
    """

    def __init__(self, device: torch.device, interval: int):
        if type(interval) is not int or interval < 0:
            raise ValueError("cuda_cache_clear_interval must be a non-negative integer")
        self.device = device
        self.interval = interval
        self._last_iteration = None
        self._clears = 0

    def reset_clock(self):
        self._last_iteration = None

    def before_collection(self, iteration: int):
        if self.device.type != "cuda" or self.interval == 0:
            return
        if self._last_iteration is not None and iteration - self._last_iteration < self.interval:
            return
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
        self._last_iteration = iteration
        self._clears += 1

    def metrics(self):
        return {"Memory/cache_clears_local": self._clears,
                "Memory/cache_clear_interval_iterations": self.interval}

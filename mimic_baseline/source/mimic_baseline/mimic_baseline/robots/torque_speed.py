"""Build and query motor torque-speed lookup tables without simulator dependencies."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


RAD_S_TO_RPM = 60.0 / (2.0 * math.pi)


def normalized_torque_speed_excess(effort, dynamic_limit, zero_speed_limit):
    """Per-joint cost in [0, 1], symmetric for driving and braking torque."""
    excess = (effort.abs() - dynamic_limit).clamp_min(0.0) / zero_speed_limit
    return excess.clamp_max(1.0).square()


@dataclass(frozen=True)
class TorqueSpeedCurve:
    """Measured non-negative motor speed and torque samples."""

    speeds_rpm: tuple[float, ...]
    torques_nm: tuple[float, ...]
    source_id: str | None = None


@dataclass(frozen=True)
class TorqueSpeedLookup:
    """Uniform torque-speed table ready for runtime indexing."""

    speed_step_rpm: float
    torque_limits_nm: tuple[float, ...]

    @property
    def max_speed_rpm(self) -> float:
        return self.speed_step_rpm * (len(self.torque_limits_nm) - 1)

    @property
    def zero_speed_torque_nm(self) -> float:
        return self.torque_limits_nm[0]


def _validate_curve(curve: TorqueSpeedCurve) -> None:
    if len(curve.speeds_rpm) != len(curve.torques_nm):
        raise ValueError(
            "A torque-speed curve must contain the same number of speeds and torques."
        )
    if len(curve.speeds_rpm) < 2:
        raise ValueError("A torque-speed curve requires at least two measured points.")
    samples = (*curve.speeds_rpm, *curve.torques_nm)
    if any(not math.isfinite(value) for value in samples):
        raise ValueError("Torque-speed samples must be finite.")
    if curve.speeds_rpm[0] < 0.0 or any(
        current >= following
        for current, following in zip(curve.speeds_rpm, curve.speeds_rpm[1:])
    ):
        raise ValueError("Motor speeds must be non-negative and strictly increasing.")
    if any(torque < 0.0 for torque in curve.torques_nm):
        raise ValueError("Motor torques must be non-negative.")


def _with_zero_speed_anchor(curve: TorqueSpeedCurve) -> TorqueSpeedCurve:
    if curve.speeds_rpm[0] == 0.0:
        return curve
    return TorqueSpeedCurve(
        speeds_rpm=(0.0, *curve.speeds_rpm),
        torques_nm=(curve.torques_nm[0], *curve.torques_nm),
        source_id=curve.source_id,
    )


def _pchip_endpoint_slope(
    adjacent_width: float,
    next_width: float,
    adjacent_secant: float,
    next_secant: float,
) -> float:
    """Compute a shape-preserving one-sided endpoint derivative."""
    slope = (
        (2.0 * adjacent_width + next_width) * adjacent_secant
        - adjacent_width * next_secant
    ) / (adjacent_width + next_width)
    if slope * adjacent_secant <= 0.0:
        return 0.0
    if adjacent_secant * next_secant < 0.0 and abs(slope) > 3.0 * abs(adjacent_secant):
        return 3.0 * adjacent_secant
    return slope


def _pchip_slopes(x: tuple[float, ...], y: tuple[float, ...]) -> tuple[float, ...]:
    """Compute Fritsch-Carlson derivatives for piecewise cubic interpolation."""
    interval_count = len(x) - 1
    widths = tuple(x[index + 1] - x[index] for index in range(interval_count))
    secants = tuple(
        (y[index + 1] - y[index]) / widths[index]
        for index in range(interval_count)
    )
    if interval_count == 1:
        return (secants[0], secants[0])

    slopes = [0.0] * len(x)
    for index in range(1, interval_count):
        left_secant = secants[index - 1]
        right_secant = secants[index]
        if left_secant * right_secant <= 0.0:
            continue
        left_width = widths[index - 1]
        right_width = widths[index]
        weight_left = 2.0 * right_width + left_width
        weight_right = right_width + 2.0 * left_width
        slopes[index] = (weight_left + weight_right) / (
            weight_left / left_secant + weight_right / right_secant
        )

    slopes[0] = _pchip_endpoint_slope(
        widths[0], widths[1], secants[0], secants[1]
    )
    slopes[-1] = _pchip_endpoint_slope(
        widths[-1], widths[-2], secants[-1], secants[-2]
    )
    return tuple(slopes)


def _evaluate_pchip(
    x: tuple[float, ...],
    y: tuple[float, ...],
    slopes: tuple[float, ...],
    query: float,
) -> float:
    interval = min(bisect_right(x, query) - 1, len(x) - 2)
    width = x[interval + 1] - x[interval]
    ratio = (query - x[interval]) / width
    ratio_squared = ratio * ratio
    ratio_cubed = ratio_squared * ratio
    return (
        (2.0 * ratio_cubed - 3.0 * ratio_squared + 1.0) * y[interval]
        + (ratio_cubed - 2.0 * ratio_squared + ratio)
        * width
        * slopes[interval]
        + (-2.0 * ratio_cubed + 3.0 * ratio_squared) * y[interval + 1]
        + (ratio_cubed - ratio_squared) * width * slopes[interval + 1]
    )


def build_conservative_torque_speed_lookup(
    curves: Sequence[TorqueSpeedCurve],
    *,
    speed_step_rpm: float = 0.25,
) -> TorqueSpeedLookup:
    """Fit measured curves and return their monotonic conservative envelope.

    Every curve is evaluated with shape-preserving piecewise cubic polynomials.
    The result takes the pointwise minimum across units, followed by a cumulative
    minimum that removes small torque increases caused by measurement noise.
    """
    if not curves:
        raise ValueError("Expected at least one measured torque-speed curve.")
    if not math.isfinite(speed_step_rpm) or speed_step_rpm <= 0.0:
        raise ValueError("Torque-speed lookup step must be positive and finite.")

    anchored_curves = []
    for curve in curves:
        _validate_curve(curve)
        anchored_curves.append(_with_zero_speed_anchor(curve))

    common_max_speed = min(curve.speeds_rpm[-1] for curve in anchored_curves)
    step_count = math.floor(common_max_speed / speed_step_rpm + 1.0e-12)
    if step_count < 1:
        raise ValueError("Torque-speed lookup step exceeds the measured speed range.")
    grid = tuple(index * speed_step_rpm for index in range(step_count + 1))

    fitted_curves = []
    for curve in anchored_curves:
        slopes = _pchip_slopes(curve.speeds_rpm, curve.torques_nm)
        fitted_curves.append(
            tuple(
                max(
                    0.0,
                    _evaluate_pchip(
                        curve.speeds_rpm,
                        curve.torques_nm,
                        slopes,
                        speed,
                    ),
                )
                for speed in grid
            )
        )

    conservative_limits = []
    previous_limit = math.inf
    for samples_at_speed in zip(*fitted_curves, strict=True):
        previous_limit = min(previous_limit, *samples_at_speed)
        conservative_limits.append(previous_limit)
    return TorqueSpeedLookup(speed_step_rpm, tuple(conservative_limits))


@dataclass(frozen=True)
class MotorTorqueSpeedProfile:
    """Measured motor curves and their precomputed conservative limits."""

    curves: tuple[TorqueSpeedCurve, ...]
    lookup: TorqueSpeedLookup
    static_effort_limit_nm: float

    @classmethod
    def from_curves(
        cls,
        curves: tuple[TorqueSpeedCurve, ...],
        *,
        speed_step_rpm: float,
        static_effort_multiplier: float,
    ) -> MotorTorqueSpeedProfile:
        """Build a profile and scale its largest measured static torque."""
        curves = tuple(curves)
        maximum_measured_torque = max(
            max(curve.torques_nm) for curve in curves
        )
        return cls(
            curves=curves,
            lookup=build_conservative_torque_speed_lookup(
                curves,
                speed_step_rpm=speed_step_rpm,
            ),
            static_effort_limit_nm=(
                static_effort_multiplier * maximum_measured_torque
            ),
        )


def lookup_torque_limit(
    joint_velocity_rad_s: torch.Tensor,
    torque_limits_with_sentinel_nm: torch.Tensor,
    *,
    speed_to_lookup_index: float,
) -> torch.Tensor:
    """Query the conservative upper-speed bin of a uniform lookup table.

    The final table element must be a zero-torque sentinel immediately above the
    measured speed range. Selecting the upper speed bin makes the approximation
    conservative and avoids per-step interpolation and range masking.
    """
    import torch

    if not torch.isfinite(joint_velocity_rad_s).all():
        bad = ~torch.isfinite(joint_velocity_rad_s)
        bad_env_ids = torch.nonzero(bad.any(dim=-1), as_tuple=False).flatten()
        raise RuntimeError(
            "Torque-speed actuator received non-finite joint velocity; "
            f"env_ids={bad_env_ids[:16].tolist()}"
        )
    lookup_index = torch.ceil(
        joint_velocity_rad_s.abs() * speed_to_lookup_index
    ).to(torch.long)
    lookup_index.clamp_max_(torque_limits_with_sentinel_nm.numel() - 1)
    return torque_limits_with_sentinel_nm[lookup_index]


def clip_effort_by_torque_speed(
    effort: torch.Tensor,
    joint_velocity_rad_s: torch.Tensor,
    torque_limits_with_sentinel_nm: torch.Tensor,
    *,
    speed_to_lookup_index: float,
) -> torch.Tensor:
    """Clamp positive and negative effort by the speed-dependent limit."""
    import torch

    dynamic_limit = lookup_torque_limit(
        joint_velocity_rad_s,
        torque_limits_with_sentinel_nm,
        speed_to_lookup_index=speed_to_lookup_index,
    )
    return torch.clamp(effort, min=-dynamic_limit, max=dynamic_limit)


class TorqueSpeedLimiter:
    """Device-resident torque-speed table with a small runtime interface."""

    __slots__ = ("_limits_with_sentinel_nm", "_speed_to_lookup_index")

    def __init__(
        self,
        *,
        speed_step_rpm: float,
        torque_limits_nm: Sequence[float],
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        import torch

        self._validate(speed_step_rpm, torque_limits_nm)
        self._limits_with_sentinel_nm = torch.tensor(
            (*torque_limits_nm, 0.0),
            dtype=dtype,
            device=device,
        )
        self._speed_to_lookup_index = RAD_S_TO_RPM / speed_step_rpm

    @staticmethod
    def _validate(
        speed_step_rpm: float,
        torque_limits_nm: Sequence[float],
    ) -> None:
        if not math.isfinite(speed_step_rpm) or speed_step_rpm <= 0.0:
            raise ValueError("Torque-speed lookup step must be positive and finite.")
        if len(torque_limits_nm) < 2:
            raise ValueError("Torque-speed lookup requires at least two limits.")
        if any(
            not math.isfinite(limit) or limit < 0.0
            for limit in torque_limits_nm
        ):
            raise ValueError("Torque-speed limits must be non-negative and finite.")
        if any(
            current < following
            for current, following in zip(
                torque_limits_nm,
                torque_limits_nm[1:],
            )
        ):
            raise ValueError(
                "Torque-speed limits must be monotonically non-increasing."
            )

    def lookup(self, joint_velocity_rad_s: torch.Tensor) -> torch.Tensor:
        """Return the conservative torque limit for each joint velocity."""
        return lookup_torque_limit(
            joint_velocity_rad_s,
            self._limits_with_sentinel_nm,
            speed_to_lookup_index=self._speed_to_lookup_index,
        )

    def clip(
        self,
        effort: torch.Tensor,
        joint_velocity_rad_s: torch.Tensor,
    ) -> torch.Tensor:
        """Clamp effort symmetrically to the measured torque-speed envelope."""
        import torch

        dynamic_limit = lookup_torque_limit(
            joint_velocity_rad_s,
            self._limits_with_sentinel_nm,
            speed_to_lookup_index=self._speed_to_lookup_index,
        )
        return torch.clamp(effort, min=-dynamic_limit, max=dynamic_limit)

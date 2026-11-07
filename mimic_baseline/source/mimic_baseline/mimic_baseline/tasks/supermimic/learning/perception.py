"""Legacy latent and clean recurrent perception for SuperMimic ERP.

The paper contract is proprioception only, so everything here is an engineering
extension of the RX-P2 task, not a paper setting.

In the legacy :class:`Perception` branch, ``z`` is a rollout state, not a command.
A goal is fixed by the reference
motion and is identical in simulation and imagination; depth is not, because
the camera rides the neck and the terrain enters and leaves the frame as the
robot moves. The world model therefore predicts ``z_{t+1}`` next to the 73-D
physical state, and imagination rolls ``z`` forward instead of replaying the
recorded frame.

Supervision has two terms, and both are needed:

* the latent term pins the predicted ``z`` to the encoder output of the *next*
  depth frame;
* the reconstruction term decodes ``z`` back to the 189-D height scan.

A latent-only loss is trivially minimised by shrinking ``z`` toward a constant,
and under end-to-end joint training nothing else would stop it. The decoder is
the anchor: ``z`` must keep enough terrain information to reproduce the scan in
metres. ``reconstruction`` therefore carries a nonzero weight whenever the
perception branch is enabled.

The clean :class:`RecurrentPerception` branch instead encodes height and depth
into current features. A GRU consumes those features and a learned WM driver,
then decodes clean height/hit and 8 m depth from one shared hidden state. Its
features and driver are not latent-residual regression targets.
"""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from mimic_baseline.tasks.supermimic.core.rotations import mat_to_6d

# Calibrated stereo ERP ray pattern (erp_cfg.ErpRayPatternCfg.image_shape).
DEPTH_SHAPE = (40, 40)
DEPTH_CHANNELS = 2
DEPTH_CLIP_M = 2.0
CLEAN_DEPTH_CLIP_M = 8.0

# erp_cfg: GridPatternCfg(resolution=0.1, size=(2.0, 0.8)) around the waist.
HEIGHT_SCAN_SIZE = (2.0, 0.8)
HEIGHT_SCAN_RESOLUTION = 0.1
HEIGHT_SCAN_SHAPE = (21, 9)
HEIGHT_SCAN_DIM = HEIGHT_SCAN_SHAPE[0] * HEIGHT_SCAN_SHAPE[1]
HEIGHT_SCAN_CLIP_M = 1.0

# Policy-side latent width agreed for the depth encoder.
LATENT_DIM = 64

# Depth is stored with a negative sentinel for "no return". A distance
# threshold cannot express validity: the tracking sensor's own ``depth_limit``
# is 8 m while the actor observation clips to 2 m, so a genuine wall at 3 m
# would look exactly like a miss. Keeping the sentinel signed makes
# ``depth < 0`` an unambiguous validity test at any range.
INVALID_DEPTH = -1.0


def height_scan_shape(
    size: tuple[float, float] = HEIGHT_SCAN_SIZE,
    resolution: float = HEIGHT_SCAN_RESOLUTION,
) -> tuple[int, int]:
    """Endpoint-inclusive (nx, ny); the native xy pattern flattens X fastest."""
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError("height_scan_size must contain two lengths in metres")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
        for value in (*size, resolution)
    ):
        raise ValueError(
            "height_scan_size and height_scan_resolution must be finite and positive"
        )
    counts = []
    for length in size:
        intervals = length / resolution
        if not math.isfinite(intervals) or intervals < 1:
            raise ValueError(
                "height_scan_size must cover at least one resolution interval per axis"
            )
        rounded = round(intervals)
        if not math.isclose(intervals, rounded, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                "height_scan_size must be an integer multiple of height_scan_resolution"
            )
        counts.append(rounded + 1)
    return counts[0], counts[1]


def height_scan_orientation(root_rotation: torch.Tensor) -> torch.Tensor:
    """World-aligned scan axes expressed in the root body frame, without Euler yaw."""
    return mat_to_6d(root_rotation.transpose(-1, -2))


def depth_features(
    depth: torch.Tensor,
    valid: torch.Tensor | None = None,
    clip_m: float = DEPTH_CLIP_M,
) -> torch.Tensor:
    """Build the two-channel encoder input used by the tracking ERP actor.

    ``depth`` is optical depth in metres with invalid pixels at
    :data:`INVALID_DEPTH`. Channel 0 is inverse normalized depth (near surfaces
    bright) zeroed outside the valid set; channel 1 marks every invalid pixel.
    Splitting the two keeps "no measurement" distinct from "surface at the far
    clip", which a single channel cannot express.

    ``valid`` defaults to the signed-sentinel test. Pass the fixed stereo mask
    explicitly when the caller also wants to exclude pixels that the calibrated
    rig never observes, as the tracking actor does.
    """
    if not 0.0 < clip_m < float("inf"):
        raise ValueError(f"clip_m must be finite and positive, got {clip_m}")
    shape = depth.shape[-2:]
    if tuple(shape) != DEPTH_SHAPE:
        raise ValueError(f"depth must end in {DEPTH_SHAPE}, got {tuple(depth.shape)}")
    measured = torch.isfinite(depth) & (depth >= 0.0)
    if valid is None:
        valid = measured
    if valid.shape != depth.shape:
        raise ValueError(
            f"valid must match depth {tuple(depth.shape)}, got {tuple(valid.shape)}"
        )
    valid = valid.bool() & measured
    normalized = (torch.where(valid, depth, 0.0) / clip_m).clamp(0.0, 1.0)
    valid_float = valid.to(depth.dtype)
    return torch.stack(((1.0 - normalized) * valid_float, 1.0 - valid_float), dim=-3)


class DepthEncoder(nn.Module):
    """40x40 two-channel ERP depth to a bounded 64-D latent.

    The convolution stack and the 576-wide penultimate layer mirror the RX-P2
    tracking actor's depth encoder, so the two tasks see the same receptive
    field and pooling geometry. ``tanh`` bounds the latent, which keeps the
    world model's residual target on a fixed scale across checkpoints.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, clip_m: float = DEPTH_CLIP_M):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if not 0.0 < clip_m < float("inf"):
            raise ValueError(f"clip_m must be finite and positive, got {clip_m}")
        self.latent_dim = int(latent_dim)
        self.clip_m = float(clip_m)
        self.conv = nn.Sequential(
            nn.Conv2d(DEPTH_CHANNELS, 16, 5, 2, 2),
            nn.ELU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.ELU(),
        )
        # 40 -> 20 -> 10 -> 5 -> 3, so the flattened map is 64 * 3 * 3.
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 64),
            nn.ELU(),
            nn.Linear(64, self.latent_dim),
        )

    def forward(
        self, depth: torch.Tensor, valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Replay stays float16 on device; convolution uses the model's dtype in
        # both training and evaluation without requiring caller-side casts.
        depth = depth.to(dtype=self.conv[0].weight.dtype)
        return torch.tanh(
            self.projection(self.conv(depth_features(depth, valid, self.clip_m)))
        )


class HeightEncoder(nn.Module):
    """Encode masked height and hit probability without discretizing soft hits."""

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        *,
        world_frame: bool = False,
        scan_dim: int = HEIGHT_SCAN_DIM,
    ):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if isinstance(scan_dim, bool) or not isinstance(scan_dim, int) or scan_dim <= 0:
            raise ValueError("scan_dim must be a positive integer")
        self.latent_dim = int(latent_dim)
        self.scan_dim = scan_dim
        self.world_frame = bool(world_frame)
        self.net = nn.Sequential(
            nn.Linear(2 * self.scan_dim + (6 if self.world_frame else 0), 128),
            nn.ELU(),
            nn.Linear(128, self.latent_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        scan: torch.Tensor,
        hit: torch.Tensor,
        orientation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if scan.shape[-1] != self.scan_dim or hit.shape != scan.shape:
            raise ValueError(
                f"scan and hit must have matching shapes ending in {self.scan_dim}, "
                f"got {tuple(scan.shape)} and {tuple(hit.shape)}"
            )
        scan = scan.to(dtype=self.net[0].weight.dtype)
        hit = hit.to(dtype=scan.dtype)
        height = (
            torch.where(torch.isfinite(scan), scan, 0.0).clamp(
                -HEIGHT_SCAN_CLIP_M, HEIGHT_SCAN_CLIP_M
            )
            / HEIGHT_SCAN_CLIP_M
        )
        features = [height * hit, hit]
        if self.world_frame:
            if orientation is None or orientation.shape != scan.shape[:-1] + (6,):
                raise ValueError(
                    "world height input requires matching 6-D scan-to-body orientation"
                )
            features.append(orientation.to(dtype=scan.dtype))
        elif orientation is not None:
            raise ValueError("legacy yaw height input does not accept scan orientation")
        return self.net(torch.cat(features, dim=-1))


class HeightScanDecoder(nn.Module):
    """Decode the latent back to the configured height scan (legacy: 189-D).

    Output is a pair: height in metres clipped to the scan's own range, and a
    hit logit per cell. Misses are not a distance, so regressing them as the
    sentinel ``-1`` would teach the decoder to fit a constant instead of
    geometry; the logit carries that information separately.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, scan_dim: int = HEIGHT_SCAN_DIM):
        super().__init__()
        if latent_dim <= 0 or scan_dim <= 0:
            raise ValueError("latent_dim and scan_dim must be positive")
        self.latent_dim = int(latent_dim)
        self.scan_dim = int(scan_dim)
        self.trunk = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )
        self.height = nn.Linear(128, self.scan_dim)
        self.hit = nn.Linear(128, self.scan_dim)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(latent)
        return (
            torch.tanh(self.height(features)) * HEIGHT_SCAN_CLIP_M,
            self.hit(features),
        )


class DepthDecoder(nn.Module):
    """Decode clean 40x40 optical depth in [0, 8] m from recurrent state."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.hidden_dim = int(hidden_dim)
        self.projection = nn.Linear(self.hidden_dim, 64 * 5 * 5)
        self.convs = nn.ModuleList(
            (
                nn.Conv2d(64, 32, 3, padding=1),
                nn.Conv2d(32, 16, 3, padding=1),
                nn.Conv2d(16, 16, 3, padding=1),
            )
        )
        self.depth = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        features = F.elu(self.projection(hidden)).reshape(-1, 64, 5, 5)
        # Avoid the costly NCHW bilinear CUDA path at large batches. Keep these
        # small images channels-last through all three scales.
        # Tensor values, parameter layout and the checkpoint format are unchanged.
        features = features.contiguous(memory_format=torch.channels_last)
        for size, conv in zip((10, 20, 40), self.convs, strict=True):
            features = F.interpolate(
                features, size=(size, size), mode="bilinear", align_corners=False
            )
            features = F.elu(conv(features))
        return torch.sigmoid(self.depth(features)).squeeze(1) * CLEAN_DEPTH_CLIP_M


class RecurrentPerception(nn.Module):
    """Height/depth features and a driver advance one shared GRU hidden state.

    Inputs are current clean measurements or previous decoded predictions.
    The caller carries hidden state and the fixed ERP validity mask through
    imagination; no measured future frame or detached feature cache is needed.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = 256,
        *,
        world_frame: bool = False,
        scan_dim: int = HEIGHT_SCAN_DIM,
    ):
        super().__init__()
        if latent_dim <= 0 or hidden_dim <= 0:
            raise ValueError("latent_dim and hidden_dim must be positive")
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.scan_dim = scan_dim
        self.world_frame = bool(world_frame)
        self.height_encoder = HeightEncoder(
            self.latent_dim, world_frame=self.world_frame, scan_dim=scan_dim
        )
        self.depth_encoder = DepthEncoder(self.latent_dim, clip_m=CLEAN_DEPTH_CLIP_M)
        self.gru = nn.GRUCell(3 * self.latent_dim, self.hidden_dim)
        self.height_decoder = HeightScanDecoder(self.hidden_dim, scan_dim)
        self.depth_decoder = DepthDecoder(self.hidden_dim)

    def encode(
        self,
        scan: torch.Tensor,
        hit: torch.Tensor,
        depth: torch.Tensor,
        orientation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.cat(
            (self.height_encoder(scan, hit, orientation), self.depth_encoder(depth)),
            dim=-1,
        )

    def initial_hidden(self, batch: int, device: torch.device | str) -> torch.Tensor:
        return torch.zeros(
            batch, self.hidden_dim, device=device, dtype=self.gru.weight_ih.dtype
        )

    def advance_hidden(
        self,
        features: torch.Tensor,
        driver: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Advance recurrent state without running reconstruction during burn-in."""
        return self.gru(torch.cat((features, driver), dim=-1), hidden)

    def step(
        self,
        features: torch.Tensor,
        driver: torch.Tensor,
        hidden: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden1 = self.advance_hidden(features, driver, hidden)
        scan1, hit_logit1 = self.height_decoder(hidden1)
        # Keep BPTT through predicted images without retaining 32 large decoder
        # activation stacks. Non-reentrant recomputation also trains the decoder
        # parameters during WM updates and preserves actor input gradients.
        depth1 = (
            checkpoint(self.depth_decoder, hidden1, use_reentrant=False)
            if torch.is_grad_enabled()
            else self.depth_decoder(hidden1)
        )
        if valid.shape != depth1.shape:
            raise ValueError(
                f"valid must match depth {tuple(depth1.shape)}, got {tuple(valid.shape)}"
            )
        depth1 = torch.where(valid.bool(), depth1, INVALID_DEPTH)
        return hidden1, scan1, hit_logit1, depth1


class Perception(nn.Module):
    """Jointly trained encoder/decoder pair plus their supervised losses.

    Parameters are shared across all ranks and averaged with the world model, so
    the encoder that produced a replay latent is never a stale copy. Replay
    stores raw depth, never ``z``: an end-to-end trained encoder changes every
    update, so a stored latent would silently mean something else by the time
    it is sampled.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        scan_dim: int = HEIGHT_SCAN_DIM,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.scan_dim = int(scan_dim)
        self.encoder = DepthEncoder(self.latent_dim)
        self.decoder = HeightScanDecoder(self.latent_dim, self.scan_dim)

    def encode(
        self, depth: torch.Tensor, valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.encoder(depth, valid)

    def decode(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(latent)


__all__ = [
    "CLEAN_DEPTH_CLIP_M",
    "DEPTH_CHANNELS",
    "DEPTH_CLIP_M",
    "DEPTH_SHAPE",
    "HEIGHT_SCAN_CLIP_M",
    "HEIGHT_SCAN_DIM",
    "HEIGHT_SCAN_SIZE",
    "HEIGHT_SCAN_RESOLUTION",
    "HEIGHT_SCAN_SHAPE",
    "INVALID_DEPTH",
    "LATENT_DIM",
    "DepthDecoder",
    "DepthEncoder",
    "HeightEncoder",
    "HeightScanDecoder",
    "Perception",
    "RecurrentPerception",
    "depth_features",
    "height_scan_orientation",
    "height_scan_shape",
]

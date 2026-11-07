"""GPU-resident RX-P2 reference motion library."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy
from pathlib import Path

import numpy as np
import torch
import yaml

from mimic_baseline.assets import ASSET_DIR

from .kinematics import RXP2Kinematics
from .rotations import mat_to_6d, quat_to_mat, yaw_mat


CONTROL_DT = 0.02
BIN_FRAMES = 25
FK_CHUNK_FRAMES = 65536
# The action interface covers the 27 body joints; the two neck joints sit at the
# end of the canonical order and are held by a positional servo, so they are not
# part of the tracked goal. Declared here rather than imported from ``mdp`` to
# avoid a cycle (``mdp`` imports this module).
ACTION_JOINTS = 27
# Per-frame goal width: dp(3) + rot6d(6) + v_b(3) + w_b(3) + q(27) + 0.1*qd(27).
GOAL_PER_FRAME = 3 + 6 + 3 + 3 + ACTION_JOINTS + ACTION_JOINTS
# Ablation control arm only: q(27) + 0.1*qd(27) + rot6d relative to the robot.
GOAL_PER_FRAME_LEGACY = 27 + 27 + 6


def _resolve_path(value: str | Path, parent: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = (
        parent / path,
        Path.cwd() / path,
        Path(ASSET_DIR).parent / path,
    )
    for ancestor in (parent, *parent.parents):
        candidates += (ancestor / path,)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item)]
    raise TypeError(f"Expected a string or sequence, got {type(value).__name__}.")


def _matches(path: Path, keywords: Sequence[str]) -> bool:
    text = str(path).lower()
    return any(keyword.lower() in text for keyword in keywords)


def _select_motion_groups(yaml_data: Mapping[str, object], yaml_path: Path) -> dict[str, list[str]]:
    groups = yaml_data.get("motion_group")
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError(f"Motion YAML has no non-empty 'motion_group': {yaml_path}")
    selected: dict[str, list[str]] = {}
    for group_name, group_value in groups.items():
        if isinstance(group_value, (str, list, tuple)):
            paths = [_resolve_path(path, yaml_path.parent) for path in _string_list(group_value)]
            included: set[Path] = set()
        elif isinstance(group_value, Mapping):
            included = set()
            for value in _string_list(group_value.get("file_name")):
                included.add(_resolve_path(value, yaml_path.parent))
            for value in _string_list(group_value.get("folder_name")):
                folder = _resolve_path(value, yaml_path.parent)
                if not folder.is_dir():
                    raise FileNotFoundError(
                        f"Motion group {group_name!r} folder does not exist: {folder}"
                    )
                included.update(path.resolve() for path in folder.rglob("*.npz") if path.is_file())
            for value in _string_list(group_value.get("wo_file_name")):
                included.discard(_resolve_path(value, yaml_path.parent))
            for value in _string_list(group_value.get("wo_folder_name")):
                folder = _resolve_path(value, yaml_path.parent)
                if folder.is_dir():
                    included.difference_update(
                        path.resolve() for path in folder.rglob("*.npz") if path.is_file()
                    )
            include_keywords = _string_list(group_value.get("include_keywords"))
            exclude_keywords = _string_list(group_value.get("exclude_keywords"))
            if include_keywords:
                included = {path for path in included if _matches(path, include_keywords)}
            if exclude_keywords:
                included = {path for path in included if not _matches(path, exclude_keywords)}
            paths = sorted(included)
        else:
            raise TypeError(f"Motion group {group_name!r} must be a mapping or path sequence.")
        if not paths:
            raise ValueError(f"Motion group {group_name!r} selected no NPZ files.")
        for path in paths:
            if path.suffix != ".npz" or not path.is_file():
                raise FileNotFoundError(f"Motion NPZ does not exist: {path}")
        selected[group_name] = [str(path) for path in paths]
    return selected


def _select_motion_paths(yaml_data: Mapping[str, object], yaml_path: Path) -> list[str]:
    return [path for paths in _select_motion_groups(yaml_data, yaml_path).values() for path in paths]


def _names(array: np.ndarray) -> list[str]:
    values = np.asarray(array).reshape(-1).tolist()
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def _scalar(data: np.lib.npyio.NpzFile, key: str) -> str:
    values = np.asarray(data[key]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Motion field {key!r} must be scalar.")
    value = values[0]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


class MotionLib:
    """Reference motions concatenated by clip into GPU tensors."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        kin: RXP2Kinematics,
        device: str | torch.device,
    ) -> None:
        if not paths:
            raise ValueError("At least one motion NPZ is required.")
        self.device = torch.device(device)
        self.kin = kin
        self.paths = tuple(str(Path(path).expanduser().resolve()) for path in paths)
        self.names = [Path(path).stem for path in self.paths]
        self.sharded = False
        self.shard_signature = ""
        self.shard_info = None
        self._load()

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        kin: RXP2Kinematics,
        device: str | torch.device,
        *,
        sharding: str = "none",
        rank: int = 0,
        world_size: int = 1,
        scene_file: str | Path | None = None,
    ) -> "MotionLib":
        if sharding not in ("none", "group"):
            raise ValueError("sharding must be none or group")
        source = Path(path).expanduser().resolve()
        payload = {}
        if source.suffix == ".npz":
            groups = {"motion": [str(source)]}
        else:
            if not source.is_file():
                raise FileNotFoundError(f"Motion YAML does not exist: {source}")
            with source.open("r", encoding="utf-8") as stream:
                payload = yaml.safe_load(stream) or {}
            if not isinstance(payload, Mapping):
                raise ValueError(f"Motion YAML root must be a mapping: {source}")
            groups = _select_motion_groups(payload, source)
        info = None
        if sharding == "group":
            from .motion_sharding import partition_motion_groups

            catalog = scene_file or payload.get("scene_file")
            if catalog is not None:
                catalog = _resolve_path(catalog, source.parent)
            paths, info = partition_motion_groups(groups, rank=rank, world_size=world_size, scene_file=catalog)
        else:
            paths = [path for group in groups.values() for path in group]
        motion = cls(paths, kin, device)
        motion.sharded = sharding == "group"
        motion.shard_info = info
        motion.shard_signature = info["signature"] if info is not None else ""
        return motion

    def _load(self) -> None:
        lengths: list[int] = []
        root_p_parts: list[np.ndarray] = []
        root_quat_parts: list[np.ndarray] = []
        root_v_parts: list[np.ndarray] = []
        root_w_parts: list[np.ndarray] = []
        joint_q_parts: list[np.ndarray] = []
        joint_qd_parts: list[np.ndarray] = []

        for path in self.paths:
            with np.load(path, allow_pickle=False) as data:
                self._validate_fields(data, path)
                file_joint_names = _names(data["joint_names"])
                file_body_names = _names(data["body_names"])
                joint_order = [
                    file_joint_names.index(name) for name in self.kin.joint_names[:27]
                ]
                root_index = file_body_names.index("base_link")

                joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
                joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
                body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
                body_quat = np.asarray(data["body_quat_w"], dtype=np.float32)
                body_lin_vel = np.asarray(data["body_lin_vel_w"], dtype=np.float32)
                body_ang_vel = np.asarray(data["body_ang_vel_w"], dtype=np.float32)
                if "quaternion_order" in data:
                    quaternion_order = _scalar(data, "quaternion_order").lower()
                    if quaternion_order == "xyzw":
                        body_quat = np.roll(body_quat, 1, axis=-1)
                    elif quaternion_order != "wxyz":
                        raise ValueError(
                            f"Unsupported quaternion_order={quaternion_order!r} in {path}."
                        )

            frame_count = int(joint_pos.shape[0])
            if frame_count < 2:
                raise ValueError(f"Motion clip must contain at least two frames: {path}")
            lengths.append(frame_count)

            joint_q = np.zeros((frame_count, len(self.kin.joint_names)), dtype=np.float32)
            joint_qd = np.zeros_like(joint_q)
            joint_q[:, :27] = joint_pos[:, joint_order]
            joint_qd[:, :27] = joint_vel[:, joint_order]
            joint_q_parts.append(joint_q)
            joint_qd_parts.append(joint_qd)
            # Copy only the base-link slices so the much larger per-body NPZ
            # arrays can be released before loading the next clip.
            root_p_parts.append(np.ascontiguousarray(body_pos[:, root_index]))
            root_quat_parts.append(np.ascontiguousarray(body_quat[:, root_index]))
            root_v_parts.append(np.ascontiguousarray(body_lin_vel[:, root_index]))
            root_w_parts.append(np.ascontiguousarray(body_ang_vel[:, root_index]))

        self.original_clip_len = torch.tensor(lengths, dtype=torch.long, device=self.device)
        padded_lengths = [((length - 2) // BIN_FRAMES + 1) * BIN_FRAMES + 1 for length in lengths]
        self.clip_len = torch.tensor(padded_lengths, dtype=torch.long, device=self.device)
        self.clip_start = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=self.device),
                self.clip_len.cumsum(0)[:-1],
            )
        )
        self.clip_end = self.clip_start + self.clip_len - 1
        # Original endpoints use the padded global indexing, for evaluation.
        self.original_clip_end = self.clip_start + self.original_clip_len - 1
        self.original_num_frames = int(sum(lengths))
        self.num_frames = int(sum(padded_lengths))
        self.clip_id = torch.repeat_interleave(
            torch.arange(len(lengths), dtype=torch.long, device=self.device),
            self.clip_len,
            output_size=self.num_frames,
        )

        # Concatenate the source arrays once on CPU and transfer them together.
        source = np.concatenate(
            (
                np.concatenate(root_p_parts),
                np.concatenate(root_quat_parts),
                np.concatenate(root_v_parts),
                np.concatenate(root_w_parts),
                np.concatenate(joint_q_parts),
                np.concatenate(joint_qd_parts),
            ),
            axis=1,
        )
        source_packed = torch.from_numpy(source).to(self.device)
        original_start = self.original_clip_len.cumsum(0) - self.original_clip_len
        offsets = torch.arange(self.num_frames, device=self.device) - self.clip_start[self.clip_id]
        original_lengths = self.original_clip_len[self.clip_id]
        source_frames = original_start[self.clip_id] + torch.minimum(offsets, original_lengths - 1)
        packed = source_packed[source_frames]
        padding = offsets >= original_lengths
        packed[:, 7:13].masked_fill_(padding[:, None], 0.0)
        packed[:, 42:71].masked_fill_(padding[:, None], 0.0)
        del source_packed, source_frames, offsets, original_lengths, padding
        self.root_p = packed[:, 0:3]
        root_quat = packed[:, 3:7]
        self.root_v = packed[:, 7:10]
        self.root_w = packed[:, 10:13]
        self.q = packed[:, 13:42]
        self.qd = packed[:, 42:71]
        load_root_R = quat_to_mat(root_quat.double())
        self.root_R = load_root_R.float()

        # Construct every body reference from exactly the loaded root/joint
        # kinematics. Double precision at load time avoids accumulating fixed
        # joint-transform error while keeping the resident tensors float32.
        load_kin = copy(self.kin)
        load_kin.off_p = self.kin.off_p.double()
        load_kin.off_R = self.kin.off_R.double()
        load_kin.axes = self.kin.axes.double()
        load_kin.q_lo = self.kin.q_lo.double()
        load_kin.q_hi = self.kin.q_hi.double()
        body_parts: list[list[torch.Tensor]] = [[], [], [], []]
        with torch.no_grad():
            for start in range(0, self.num_frames, FK_CHUNK_FRAMES):
                end = min(start + FK_CHUNK_FRAMES, self.num_frames)
                body = load_kin.forward(
                    self.root_p[start:end].double(),
                    load_root_R[start:end],
                    self.root_v[start:end].double(),
                    self.root_w[start:end].double(),
                    self.q[start:end].double(),
                    self.qd[start:end].double(),
                )
                for parts, value in zip(body_parts, body, strict=True):
                    parts.append(value.float())
        self.body_p, self.body_R, self.body_v, self.body_w = (
            torch.cat(parts) for parts in body_parts
        )
        self.root_gz = self.root_R[..., 2, 2]
        # Reference root velocities in the root body frame, precomputed once so
        # the per-step goal is a pure gather plus a small matmul.
        root_R_transpose = self.root_R.transpose(-1, -2)
        self.root_v_local = (root_R_transpose @ self.root_v.unsqueeze(-1)).squeeze(-1)
        self.root_w_local = (root_R_transpose @ self.root_w.unsqueeze(-1)).squeeze(-1)
        self._build_bins()

    @staticmethod
    def _validate_fields(data: np.lib.npyio.NpzFile, path: str) -> None:
        required = {
            "fps",
            "robot_name",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "body_names",
            "joint_names",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"Motion file {path} is missing fields: {missing}.")
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if not np.isclose(fps, 50.0):
            raise ValueError(f"Motion file {path} has fps={fps}, expected 50.")
        if not _scalar(data, "robot_name").startswith("mondo_rx_p2"):
            raise ValueError(f"Motion file {path} is not an RX-P2 motion.")

    def _build_bins(self) -> None:
        bin_counts = (self.clip_len - 1) // BIN_FRAMES
        bin_offsets = bin_counts.cumsum(0) - bin_counts
        self.num_bins = (self.num_frames - self.clip_len.numel()) // BIN_FRAMES
        self.bin_clip = torch.repeat_interleave(
            torch.arange(self.clip_len.numel(), device=self.device), bin_counts,
            output_size=self.num_bins,
        )
        self.bin_start = self.clip_start[self.bin_clip] + BIN_FRAMES * (
            torch.arange(self.num_bins, device=self.device) - bin_offsets[self.bin_clip]
        )
        local_frame = torch.arange(self.num_frames, device=self.device) - self.clip_start[self.clip_id]
        # A clip's final state has no outgoing transition and belongs to its
        # last bin for lookup purposes; it is never a valid sampled start.
        self.frame_bin = bin_offsets[self.clip_id] + torch.minimum(
            local_frame // BIN_FRAMES, bin_counts[self.clip_id] - 1
        )

    def ref_state(
        self, f: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]

    def future_frames(self, f: torch.Tensor, K: int) -> torch.Tensor:
        if K < 1:
            raise ValueError(f"K must be positive, got {K}.")
        end = self.clip_end[self.clip_id[f]]
        offsets = torch.arange(1, K + 1, device=f.device)
        return torch.minimum(f[..., None] + offsets, end[..., None])

    def goal(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Return the pure-reference goal for the K frames after ``f``.

        The paper defines ``g_t`` as read from the reference motion, so no robot
        quantity may enter: the conditioning is then identical in simulation and
        inside the world model's imagination, where a robot-state-dependent goal
        would inherit the model's own prediction error.

        Per-frame layout (69): heading-local position delta ``dp`` (3),
        heading-local rot6d of the future root (6), root velocity in the root
        body frame (3+3), the 27 actuated joint positions, and their scaled
        velocities (27). Root height is deliberately excluded: height is exactly
        the quantity the termination term already penalises, so repeating it in
        the goal would give the actor a shortcut that does not transfer.
        """
        future = self.future_frames(f, K)
        heading_transpose = yaw_mat(self.root_R[f]).transpose(-1, -2).unsqueeze(-3)
        features = torch.cat(
            (
                (
                    heading_transpose
                    @ (self.root_p[future] - self.root_p[f].unsqueeze(-2)).unsqueeze(-1)
                ).squeeze(-1),
                mat_to_6d(heading_transpose @ self.root_R[future]),
                self.root_v_local[future],
                self.root_w_local[future],
                self.q[future][..., :ACTION_JOINTS],
                self.qd[future][..., :ACTION_JOINTS] * 0.1,
            ),
            dim=-1,
        )
        return features.reshape(f.shape + (GOAL_PER_FRAME * K,))

    def goal_state_relative(
        self, f: torch.Tensor, K: int, R_actual: torch.Tensor
    ) -> torch.Tensor:
        """Legacy control arm: 60-D goal anchored on the robot's own rotation.

        Kept only so the ablation can isolate the reference-only change. It must
        not be used for production training.
        """
        future = self.future_frames(f, K)
        relative_rotation = (
            R_actual.transpose(-1, -2).unsqueeze(-3) @ self.root_R[future]
        )
        features = torch.cat(
            (
                self.q[future][..., :27],
                self.qd[future][..., :27] * 0.1,
                mat_to_6d(relative_rotation),
            ),
            dim=-1,
        )
        return features.reshape(f.shape + (GOAL_PER_FRAME_LEGACY * K,))


__all__ = ["BIN_FRAMES", "CONTROL_DT", "MotionLib"]

"""G1 metadata for the original ManagerBased SuperMimic implementation.

Joint order is the LAFAN1 retarget CSV order. FK uses the original project's
URDF parser and differentiable tree recursion, including fixed links.
"""

import torch

from mimic_baseline.tasks.supermimic.core.kinematics import RXP2Kinematics
from mimic_baseline.tasks.supermimic.core.rotations import axis_angle_to_mat

G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
G1_TRACKED_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


class G1Kinematics(RXP2Kinematics):
    """G1 configuration of the existing URDF FK, with all 29 joints actuated."""

    recovery_required_bodies = ("pelvis_contour_link", "torso_link")

    def __init__(self, urdf_path, device):
        super().__init__(
            urdf_path,
            device,
            joint_names=G1_JOINT_NAMES,
            tracked_body_names=G1_TRACKED_BODY_NAMES,
            root_body_name="pelvis",
            action_dim=29,
        )

        index = self.tracked_body_names.index
        self.recovery_body_indices = (
            index("torso_link"),
            (index("left_ankle_roll_link"), index("right_ankle_roll_link")),
            (index("left_wrist_yaw_link"), index("right_wrist_yaw_link")),
        )

    def _initialize_fast_topology(self):
        # The parser orders links breadth-first. Cache only integer topology,
        # retaining the original tensors as the single source of transforms.
        levels = [[0]]
        while True:
            previous = {index: slot for slot, index in enumerate(levels[-1])}
            children = [i for i, parent in enumerate(self.parent) if parent in previous]
            if not children:
                break
            levels.append(children)
        order = [index for level in levels for index in level]
        if order != list(range(len(self.body_names))):
            raise ValueError("Fast G1 FK requires breadth-first body ordering.")
        self._fast_levels = tuple(
            (
                torch.tensor(level, device=self.device),
                torch.tensor([levels[depth - 1].index(self.parent[i]) for i in level],
                             device=self.device),
            )
            for depth, level in enumerate(levels) if depth
        )
        self._fast_tracked = torch.tensor(self.tracked_idx, device=self.device)

    def forward_fast(self, p, R, v, w, q, qd):
        """Optional depth-batched FK with the reference forward contract.

        This is opt-in: inherited ``forward`` remains the default. No tensors
        are detached or written in-place, preserving BPTT gradients. Fixed
        joints use identity rotations and zero angular velocity increments.
        """
        if q.shape[-1] != len(self.joint_names) or qd.shape != q.shape:
            raise ValueError(
                f"Expected q and qd ending in {len(self.joint_names)}, got "
                f"{tuple(q.shape)} and {tuple(qd.shape)}."
            )
        if not hasattr(self, "_fast_levels"):
            self._initialize_fast_topology()
        angles = torch.where(self._pose_moving, q[..., self._pose_joint_indices], 0.)
        rates = torch.where(self._pose_moving, qd[..., self._pose_joint_indices], 0.)
        joint_rotations = axis_angle_to_mat(self.axes, angles)
        positions, rotations = [p.unsqueeze(-2)], [R.unsqueeze(-3)]
        velocities, angular_velocities = [v.unsqueeze(-2)], [w.unsqueeze(-2)]
        for links, parents in self._fast_levels:
            parent_p = positions[-1].index_select(-2, parents)
            parent_R = rotations[-1].index_select(-3, parents)
            parent_v = velocities[-1].index_select(-2, parents)
            parent_w = angular_velocities[-1].index_select(-2, parents)
            link_p = parent_p + (parent_R @ self.off_p[links, :, None]).squeeze(-1)
            fixed_R = parent_R @ self.off_R[links]
            link_R = fixed_R @ joint_rotations.index_select(-3, links)
            world_axis = (fixed_R @ self.axes[links, :, None]).squeeze(-1)
            link_w = parent_w + world_axis * rates.index_select(-1, links).unsqueeze(-1)
            link_v = parent_v + torch.cross(parent_w, link_p - parent_p, dim=-1)
            positions.append(link_p)
            rotations.append(link_R)
            velocities.append(link_v)
            angular_velocities.append(link_w)
        return tuple(
            torch.cat(values, dim=dim).index_select(dim, self._fast_tracked)
            for values, dim in ((positions, -2), (rotations, -3),
                                (velocities, -2), (angular_velocities, -2))
        )


class G1BatchedKinematics(G1Kinematics):
    """Opt-in batched FK whose method remains correctly bound after copying.

    MotionLib shallow-copies kinematics before casting its URDF transforms to
    double precision. A class method alias follows the copied instance, unlike
    assigning an already-bound method to an instance attribute.
    """

    forward = G1Kinematics.forward_fast


class G1LeanKinematics(G1Kinematics):
    """Opt-in depth batches restricted to ancestors of tracked bodies.

    Unlike the evaluation batch path, this never concatenates all link states
    before selecting tracked bodies. Each tracked state is selected directly
    from its level, and unused leaf branches never enter the autograd graph.
    Static-transform contractions keep URDF constants shared across states,
    rather than materializing per-state copies for broadcast matrix products.
    Only integer topology is cached, preserving copied/double URDF transforms.
    """

    def _initialize_lean_topology(self):
        required = set(self.tracked_idx)
        for link in self.tracked_idx:
            while self.parent[link] >= 0:
                link = self.parent[link]
                required.add(link)
        previous = [0]
        levels = []
        while True:
            links = [i for i, parent in enumerate(self.parent) if i in required and parent in previous]
            if not links:
                break
            moving = [slot for slot, link in enumerate(links) if self.joint_idx[link] >= 0]
            parent_slots = [previous.index(self.parent[i]) for i in links]
            parent_view = None
            if parent_slots == list(range(parent_slots[0], parent_slots[0] + len(parent_slots))):
                parent_view = (parent_slots[0], len(parent_slots))
            elif len(set(parent_slots)) == 1:
                parent_view = (parent_slots[0], 1)
            def integer(values):
                return torch.tensor(values, dtype=torch.long, device=self.device)
            levels.append((
                integer(links), integer(parent_slots), parent_view,
                integer(moving), integer([self.joint_idx[links[slot]] for slot in moving]),
                tuple((slot, self.tracked_idx.index(link)) for slot, link in enumerate(links)
                      if link in self.tracked_idx),
            ))
            previous = links
        self._lean_levels = tuple(levels)

    def forward(self, p, R, v, w, q, qd):
        if q.shape[-1] != len(self.joint_names) or qd.shape != q.shape:
            raise ValueError(
                f"Expected q and qd ending in {len(self.joint_names)}, got "
                f"{tuple(q.shape)} and {tuple(qd.shape)}."
            )
        if not hasattr(self, '_lean_levels'):
            self._initialize_lean_topology()
        current = (p.unsqueeze(-2), R.unsqueeze(-3), v.unsqueeze(-2), w.unsqueeze(-2))
        tracked = [None] * len(self.tracked_idx)
        if 0 in self.tracked_idx:
            tracked[self.tracked_idx.index(0)] = (p, R, v, w)
        for links, parents, parent_view, moving, joints, selected in self._lean_levels:
            gathered = []
            for tensor, dim in zip(current, (-2, -3, -2, -2)):
                if parent_view is None:
                    parent_tensor = tensor.index_select(dim, parents)
                else:
                    start, count = parent_view
                    parent_tensor = (tensor if start == 0 and count == tensor.shape[dim]
                                     else tensor.narrow(dim, start, count))
                    if count != parents.numel():
                        shape = list(parent_tensor.shape)
                        shape[dim] = parents.numel()
                        parent_tensor = parent_tensor.expand(shape)
                gathered.append(parent_tensor)
            parent_p, parent_R, parent_v, parent_w = gathered
            offset = torch.einsum("...kij,kj->...ki", parent_R, self.off_p[links])
            link_p = parent_p + offset
            fixed_R = torch.einsum("...kij,kjl->...kil", parent_R, self.off_R[links])
            if moving.numel():
                all_moving = moving.numel() == links.numel()
                moving_R = fixed_R if all_moving else fixed_R.index_select(-3, moving)
                axes = self.axes[links] if all_moving else self.axes[links[moving]]
                joint_R = axis_angle_to_mat(axes, q.index_select(-1, joints))
                rotated = moving_R @ joint_R
                world_axis = torch.einsum("...kij,kj->...ki", moving_R, axes)
                increment = world_axis * qd.index_select(-1, joints).unsqueeze(-1)
                if all_moving:
                    link_R = rotated
                    link_w = parent_w + increment
                else:
                    link_R = fixed_R.index_copy(-3, moving, rotated)
                    link_w = parent_w.index_copy(-2, moving,
                                                parent_w.index_select(-2, moving) + increment)
            else:
                link_R, link_w = fixed_R, parent_w
            # Retain reference subtraction semantics (including float rounding).
            link_v = parent_v + torch.cross(parent_w, link_p - parent_p, dim=-1)
            current = (link_p, link_R, link_v, link_w)
            for slot, output_index in selected:
                tracked[output_index] = tuple(tensor.select(dim, slot)
                                              for tensor, dim in zip(current, (-2, -3, -2, -2)))
        return tuple(torch.stack(fields, dim=dim)
                     for fields, dim in zip(zip(*tracked), (-2, -3, -2, -2)))


class G1TrainingKinematics(G1LeanKinematics):
    """Opt-in differentiable FK with bounded recomputation workspace.

    The lean depth-batched implementation saves per-link intermediates when
    called on flattened policy horizons. Checkpoint each bounded chunk instead:
    only the six inputs survive forward, and autograd recomputes one chunk's
    intermediates at a time. Outputs and their gradients still scale with the
    full input batch; ``chunk_size`` bounds FK workspace, not total training
    memory. No policy samples or time steps are dropped or detached.

    This trades a second FK forward during backward for reduced activation
    storage. The reference and evaluation implementations remain unchanged.
    """

    def __init__(self, urdf_path, device, *, chunk_size=4096):
        super().__init__(urdf_path, device)
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self.chunk_size = chunk_size

    def forward(self, p, R, v, w, q, qd):
        from torch.utils.checkpoint import checkpoint

        if q.shape[-1] != len(self.joint_names) or qd.shape != q.shape:
            raise ValueError(
                f"Expected q and qd ending in {len(self.joint_names)}, got "
                f"{tuple(q.shape)} and {tuple(qd.shape)}."
            )
        inputs = (p, R, v, w, q, qd)
        tails = ((3,), (3, 3), (3,), (3,), (len(self.joint_names),), (len(self.joint_names),))
        batch = torch.broadcast_shapes(*(x.shape[:-len(tail)] for x, tail in zip(inputs, tails)))
        flattened = tuple(x.expand(*batch, *tail).reshape(-1, *tail)
                          for x, tail in zip(inputs, tails))
        count = flattened[0].shape[0]
        # Initialize integer-only topology outside recomputation. Float transforms
        # are always read from this instance (including MotionLib's double copy).
        if not hasattr(self, '_lean_levels'):
            self._initialize_lean_topology()
        outputs = []
        for start in range(0, max(count, 1), self.chunk_size):
            chunk = tuple(x[start:start + self.chunk_size] for x in flattened)
            if torch.is_grad_enabled() and any(x.requires_grad for x in chunk):
                result = checkpoint(super().forward, *chunk, use_reentrant=False,
                                    preserve_rng_state=False)
            else:
                result = super().forward(*chunk)
            outputs.append(result)
        return tuple(torch.cat(parts, dim=0).reshape(*batch, *parts[0].shape[1:])
                     for parts in zip(*outputs))

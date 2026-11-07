"""Episode-local domain randomization for the standalone RX-P2 simulator."""

from __future__ import annotations

import operator

import torch


TRAIN_FRICTION_RANGE = (0.3, 1.2)
TRAIN_CALIBRATION_RANGE = (-0.01, 0.01)
TRAIN_COM_RANGE = ((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05))
EVAL_FRICTION = 1.0
EVAL_CALIBRATION = 0.0
EVAL_COM = (0.0, 0.0, 0.0)


def _to_torch(value) -> torch.Tensor:
    """Bridge a Warp array to a torch tensor for the one-time cache."""
    import warp as wp

    return wp.to_torch(value)


def _from_torch(tensor: torch.Tensor, *, dtype):
    """Bridge a CPU torch tensor to Warp for PhysX's CPU-only material API."""
    import warp as wp

    return wp.from_torch(tensor, dtype=dtype)


class EpisodeRandomization:
    """Resample the paper's three context dimensions for selected episodes.

    This helper must be constructed after the interactive scene and robot are
    initialized, but before the first training reset. Construction captures the
    current material properties and torso COM pose once; it assumes those values
    are the nominal scene state and does not write them to PhysX.

    ``xi[:, 0]`` is friction, ``xi[:, 1:30]`` is the 29-joint calibration, and
    ``xi[:, 30:33]`` is the torso COM offset. Every call replaces those values
    for ``env_ids``. The COM is always rebuilt from the cached nominal pose, so
    repeated resets cannot accumulate offsets.
    """

    def __init__(
        self,
        robot,
        xi: torch.Tensor,
        torso_id: int,
        eval_mode: bool = False,
        physics_manager=None,
    ) -> None:
        if not isinstance(xi, torch.Tensor):
            raise TypeError(
                f"xi must be a torch.Tensor, got {type(xi).__name__}."
            )
        if xi.ndim != 2 or xi.shape[1] != 33:
            raise ValueError(
                f"xi must have shape (num_envs, 33), got {tuple(xi.shape)}."
            )
        try:
            torso_id = operator.index(torso_id)
        except TypeError as exc:
            raise TypeError("torso_id must be an integer body index.") from exc

        self.robot = robot
        self.xi = xi
        self.torso_id = torso_id
        self.eval_mode = bool(eval_mode)
        self.num_envs = xi.shape[0]
        self.device = xi.device

        self._newton_manager = physics_manager
        if physics_manager is not None:
            model = physics_manager.get_model()
            self._newton_mu = _to_torch(robot.root_view.get_attribute("shape_material_mu", model)[:, 0])
            # MuJoCo combines contact friction using max. Ground must not mask
            # the original robot friction range [0.3, 1.2].
            shape_body = _to_torch(model.shape_body)
            _to_torch(model.shape_material_mu)[shape_body < 0] = TRAIN_FRICTION_RANGE[0]
        else:
            self._ov_material_type = None
            if hasattr(robot.root_view, "get_material_properties"):
                materials = robot.root_view.get_material_properties()
            else:
                from isaaclab_ov import tensor_types
                self._ov_material_type = tensor_types.SHAPE_FRICTION_AND_RESTITUTION
                materials = robot.root_view.get_attribute(self._ov_material_type)
            material_properties = _to_torch(materials)
            if not isinstance(material_properties, torch.Tensor):
                raise TypeError("Material properties must bridge to a torch.Tensor.")
            if (
                material_properties.ndim != 3
                or material_properties.shape[0] != self.num_envs
                or material_properties.shape[2] != 3
            ):
                raise ValueError(
                    "Material properties must have shape "
                    "(num_envs, max_shapes, 3), got "
                    f"{tuple(material_properties.shape)}."
                )
            # PhysX's material setter consumes a full CPU tensor plus selected
            # environment indices.
            self._material_properties = (
                material_properties.detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                .clone()
            )

        nominal_com = robot.data.body_com_pose_b.torch[:, torso_id]
        if nominal_com.ndim != 2 or nominal_com.shape[0] != self.num_envs:
            raise ValueError(
                "Torso COM must have shape (num_envs, pose_dim), got "
                f"{tuple(nominal_com.shape)}."
            )
        if nominal_com.shape[1] < 3:
            raise ValueError("Torso COM pose must contain at least xyz.")
        self._nominal_com = nominal_com.detach().clone()

    def resample(self, env_ids: torch.Tensor) -> None:
        """Replace all context values and physics state for selected episodes."""
        if env_ids.numel() == 0:
            return

        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        count = env_ids.numel()

        if self.eval_mode:
            self.xi[env_ids, 0] = EVAL_FRICTION
            self.xi[env_ids, 1:30] = EVAL_CALIBRATION
            self.xi[env_ids, 30] = EVAL_COM[0]
            self.xi[env_ids, 31] = EVAL_COM[1]
            self.xi[env_ids, 32] = EVAL_COM[2]
        else:
            self.xi[env_ids, 0] = torch.empty(
                count, dtype=self.xi.dtype, device=self.device
            ).uniform_(*TRAIN_FRICTION_RANGE)
            self.xi[env_ids, 1:30] = torch.empty(
                (count, 29), dtype=self.xi.dtype, device=self.device
            ).uniform_(*TRAIN_CALIBRATION_RANGE)
            com_offsets = torch.empty(
                (count, 3), dtype=self.xi.dtype, device=self.device
            ).uniform_()
            com_offsets[:, 0].mul_(
                TRAIN_COM_RANGE[0][1] - TRAIN_COM_RANGE[0][0]
            ).add_(TRAIN_COM_RANGE[0][0])
            com_offsets[:, 1].mul_(
                TRAIN_COM_RANGE[1][1] - TRAIN_COM_RANGE[1][0]
            ).add_(TRAIN_COM_RANGE[1][0])
            com_offsets[:, 2].mul_(
                TRAIN_COM_RANGE[2][1] - TRAIN_COM_RANGE[2][0]
            ).add_(TRAIN_COM_RANGE[2][0])
            self.xi[env_ids, 30:33] = com_offsets

        self._apply_material(env_ids)
        self._apply_torso_com(env_ids)

    def _apply_material(self, env_ids: torch.Tensor) -> None:
        """Write selected friction rows through PhysX's CPU material bridge."""
        if self._newton_manager is not None:
            from newton import ModelFlags
            self._newton_mu[env_ids] = self.xi[env_ids, :1]
            self._newton_manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
            return
        import warp as wp

        cpu_ids = env_ids.detach().to(device="cpu", dtype=torch.int32)
        friction = self.xi[env_ids, 0].detach().to(
            device="cpu", dtype=torch.float32
        )
        selected_friction = friction[:, None]
        self._material_properties[cpu_ids.long(), :, 0] = selected_friction
        self._material_properties[cpu_ids.long(), :, 1] = selected_friction
        values = _from_torch(self._material_properties.contiguous(), dtype=wp.float32)
        indices = _from_torch(cpu_ids, dtype=wp.int32)
        if self._ov_material_type is None:
            self.robot.root_view.set_material_properties(values, indices)
        else:
            self.robot.root_view.set_attribute(self._ov_material_type, values, indices=indices)

    def _apply_torso_com(self, env_ids: torch.Tensor) -> None:
        """Rebase selected torso COM positions on the cached nominal pose."""
        coms = self._nominal_com[env_ids].clone()
        coms[:, :3].add_(self.xi[env_ids, 30:33])
        self.robot.set_coms_index(
            coms=coms[:, None, :],
            body_ids=[self.torso_id],
            env_ids=env_ids,
        )


__all__ = [
    "EpisodeRandomization",
    "EVAL_CALIBRATION",
    "EVAL_COM",
    "EVAL_FRICTION",
    "TRAIN_CALIBRATION_RANGE",
    "TRAIN_COM_RANGE",
    "TRAIN_FRICTION_RANGE",
]

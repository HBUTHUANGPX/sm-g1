"""G1 configuration of the original SuperMimic ManagerBased environment."""

from isaaclab.utils.configclass import configclass
from .env_managed import SuperMimicEnvCfg


@configclass
class G1SuperMimicEnvCfg(SuperMimicEnvCfg):
    self_collisions: bool = True
    backend: str = "ovphysx"
    robot_name: str = "g1"
    action_dim: int = 29
    torso_body_name: str = "torso_link"
    actuator_type: str = "implicit"
    min_delay: int = 0
    max_delay: int = 0

    def __post_init__(self):
        if self.recovery_enabled:
            raise ValueError("G1 recovery extension is not validated; use the original flat task")
        super().__post_init__()
        if self.backend == "ovphysx":
            from isaaclab_ov.physics import OvPhysxCfg

            self.sim.physics = OvPhysxCfg()
        elif self.backend == "newton":
            from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg

            self.sim.physics = NewtonCfg(
                solver_cfg=MJWarpSolverCfg(
                    njmax=500, nconmax=128, cone="pyramidal", integrator="implicitfast", use_mujoco_contacts=True
                ),
                num_substeps=1,
            )
            self.scene.robot.spawn.variants = {"Physics": "mujoco"}
            self.scene.env_spacing = 0.0
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

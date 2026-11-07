"""G1 configuration of the original SuperMimic ManagerBased environment."""

from isaaclab.utils.configclass import configclass
from mimic_baseline.tasks.supermimic.envs.configs.base import SuperMimicEnvCfg


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
            # Custom delay buffers execute in Lab on every physics tick.
            self.sim.use_newton_actuators = self.actuator_type == "implicit"
            self.scene.robot.spawn.variants = {"Physics": "mujoco"}
            self.scene.env_spacing = 0.0
        else:
            raise ValueError(f"Unknown backend: {self.backend}")


@configclass
class G1RecoveryEnvCfg(G1SuperMimicEnvCfg):
    recovery_enabled: bool = True
    actuator_type: str = "delayed_pd"
    min_delay: int = 0
    max_delay: int = 4

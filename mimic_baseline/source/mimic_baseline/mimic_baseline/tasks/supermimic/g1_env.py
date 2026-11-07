"""G1 configuration of the original SuperMimic ManagerBased environment."""

from isaaclab.utils.configclass import configclass
from .env_managed import SuperMimicEnvCfg


@configclass
class G1SuperMimicEnvCfg(SuperMimicEnvCfg):
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
        from isaaclab_ov.physics import OvPhysxCfg

        self.sim.physics = OvPhysxCfg()

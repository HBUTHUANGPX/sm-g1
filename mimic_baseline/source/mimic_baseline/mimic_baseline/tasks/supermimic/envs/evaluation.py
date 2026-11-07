"""Evaluation-only specialization of the original ManagerBased G1 environment."""

from isaaclab.managers import TerminationManager
from isaaclab.utils.configclass import configclass
from .manager_based import SuperMimicRLEnv
from .configs.g1 import G1SuperMimicEnvCfg
from ..evaluation.profiles import configure_evaluation


@configclass
class G1EvaluationEnvCfg(G1SuperMimicEnvCfg):
    evaluation_profile: str = "strict"
    evaluation_clip_steps: int = 1


class G1EvaluationEnv(SuperMimicRLEnv):
    def __init__(self, cfg, **kwargs):
        configure_evaluation(
            cfg, cfg.evaluation_profile, longest_clip_steps=cfg.evaluation_clip_steps
        )
        super().__init__(cfg, **kwargs)

    @property
    def evaluation_profile(self):
        return self.cfg.evaluation_profile

    def load_managers(self):
        super().load_managers()
        if self.cfg.evaluation_profile == "full_clean":
            # An empty native manager disables BOTH termination and truncation.
            # Do not just ignore done flags after the simulator has autoreset.
            self.cfg.terminations = {}
            self.termination_manager = TerminationManager({}, self)

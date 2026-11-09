"""Evaluation-only specialization of the original ManagerBased G1 environment."""

from isaaclab.managers import TerminationManager
from isaaclab.utils.configclass import configclass

from ..evaluation.profiles import configure_evaluation
from .configs.g1 import G1SuperMimicEnvCfg
from .manager_based import SuperMimicRLEnv


@configclass
class G1EvaluationEnvCfg(G1SuperMimicEnvCfg):
    evaluation_profile: str = "strict"
    evaluation_clip_steps: int = 1
    evaluation_execution: str = "reference"


class G1EvaluationEnv(SuperMimicRLEnv):
    def __init__(self, cfg, **kwargs):
        configure_evaluation(
            cfg, cfg.evaluation_profile, longest_clip_steps=cfg.evaluation_clip_steps
        )
        if cfg.evaluation_execution == "grouped":
            from .evaluation_actuators import group_evaluation_actuators

            group_evaluation_actuators(cfg)
        elif cfg.evaluation_execution != "reference":
            raise ValueError("Unknown evaluation execution mode")
        super().__init__(cfg, **kwargs)

    def state(self):
        # Native evaluation suppresses autoresets, so physics does not change
        # between the post-step snapshot and observation-manager reads. Never
        # reuse it outside step: explicit resets must see the newly written pose.
        if self._stepping and self._post_state is not None:
            return self._post_state
        return super().state()

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

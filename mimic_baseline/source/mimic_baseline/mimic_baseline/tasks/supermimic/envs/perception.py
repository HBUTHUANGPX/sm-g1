"""Original ManagerBased perception responsibilities."""

from __future__ import annotations

import math

from isaaclab.utils.configclass import configclass

from mimic_baseline.tasks.supermimic.envs.configs.base import (
    SuperMimicEnvCfg,
    NECK_PITCH_DOWN_29DEG,
)
from mimic_baseline.tasks.supermimic.envs.manager_based import SuperMimicRLEnv


@configclass
class SuperMimicErpEnvCfg(SuperMimicEnvCfg):
    neck_pitch: float = NECK_PITCH_DOWN_29DEG
    erp_depth_noise_m: float = 0.05


class SuperMimicErpRLEnv(SuperMimicRLEnv):
    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            from mimic_baseline.tasks.supermimic.erp import close_raycast_backends

            try:
                close_raycast_backends(self)
            except Exception:
                pass
            raise

    def close(self):
        from mimic_baseline.tasks.supermimic.erp import close_raycast_backends

        try:
            close_raycast_backends(self)
        finally:
            # Kit can tick during shutdown while another UI owner still holds
            # this controller. Unsubscribe before the base close deletes scene.
            controller = getattr(self, "viewport_camera_controller", None)
            handle = getattr(controller, "_viewport_camera_update_handle", None)
            if handle is not None:
                handle.unsubscribe()
                controller._viewport_camera_update_handle = None
            super().close()

    @property
    def camera_geometry(self):
        """Native RF mesh provider, forwarded by SuperMimicTrainingEnv."""
        from mimic_baseline.tasks.supermimic.geometry import ErpCameraGeometry

        geometry = getattr(self, "_supermimic_camera_geometry", None)
        if geometry is None:
            geometry = self._supermimic_camera_geometry = ErpCameraGeometry(
                self.scene.sensors["erp_camera"],
                self.scene.terrain,
                self.motion,
            )
        return geometry

    @property
    def camera_transport(self):
        """The real capture transport, also exposed through the training adapter."""
        from mimic_baseline.tasks.supermimic.erp import _camera_transport

        return _camera_transport(self)

    def _configure_scene(self, cfg):
        from mimic_baseline.tasks.supermimic.erp import configure_erp_scene

        configure_erp_scene(cfg, self.motion)

    def bind_motion_terrain(self, motion_ids, env_ids):
        from mimic_baseline.tasks.supermimic.erp import bind_terrain

        bind_terrain(self, motion_ids, env_ids)

    def perception(self):
        from mimic_baseline.tasks.supermimic.erp import capture_perception

        return capture_perception(self)

    def _capture_start_perception(self):
        if self.cfg.clean_perception:
            self.transition.update(
                {
                    key: value.clone()
                    for key, value in self.perception().items()
                    if key
                    in (
                        "depth",
                        "scan",
                        "hit",
                        "clean_depth",
                        "camera_frame",
                        "camera_lag",
                    )
                }
            )
            return
        self.transition.update(
            {
                key: value.clone()
                for key, value in self.perception().items()
                if key in ("depth", "scan", "hit")
            }
        )

    def _capture_end_perception(self):
        if self.cfg.clean_perception:
            self.transition.update(
                {
                    key + "1": value.clone()
                    for key, value in self.perception().items()
                    if key in ("depth", "scan", "hit", "clean_depth")
                }
            )
            return
        self.transition.update(
            {
                key + "1": value.clone()
                for key, value in self.perception().items()
                if key in ("depth", "scan", "hit")
            }
        )

    def _reset_sensors(self, env_ids):
        from mimic_baseline.tasks.supermimic.erp import reset_perception

        reset_perception(self, env_ids)

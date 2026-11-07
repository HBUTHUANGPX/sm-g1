"""Evaluation protocol constants."""

SUITES = ("tracking", "standing_push")


PUSH_UPRIGHT_THRESHOLD = 0.8


PUSH_ROOT_HEIGHT_TOLERANCE = 0.25


PUSH_FINAL_DWELL_STEPS = 25


METRIC_KEYS = (
    "root_xy_rmse",
    "root_z_rmse",
    "root_rotation_rmse_rad",
    "joint_rmse",
    "body_position_rmse",
    "body_orientation_rmse_rad",
)

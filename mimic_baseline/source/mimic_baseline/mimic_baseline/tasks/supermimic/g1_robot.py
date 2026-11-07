"""G1 asset/control configuration for the existing manager environment.

Use source URDF joint limits. Gains below are an explicit initial engineering
configuration, not a claim of hardware calibration or policy convergence.
The environment and trainer stay in the original project.
"""

import xml.etree.ElementTree as ET
from .g1 import G1_JOINT_NAMES


def make_g1_robot_cfg(usd_path, urdf_path, *, actuator_type="implicit", min_delay=0, max_delay=0):
    if not usd_path or not urdf_path:
        raise ValueError("G1 requires explicit robot_usd and robot_urdf paths")
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg, PhysxArticulationCfg

    if actuator_type not in ("implicit", "delayed_pd"):
        raise ValueError("G1 supports implicit or delayed_pd without TN")
    if min_delay < 0 or max_delay < min_delay:
        raise ValueError("Expected 0 <= min_delay <= max_delay")
    if actuator_type == "implicit" and max_delay:
        raise ValueError("Implicit G1 actuators do not implement delay")
    from .actuators import G1DelayedPDActuatorCfg

    joints = {j.attrib["name"]: j for j in ET.parse(urdf_path).getroot().findall("joint")}
    actuators = {}
    scales = []
    defaults = {}
    for name in G1_JOINT_NAMES:
        limit = joints[name].find("limit")
        effort = float(limit.attrib["effort"])
        velocity = float(limit.attrib["velocity"])
        if "knee" in name:
            kp, kd, default = 200.0, 5.0, 0.3
        elif "ankle" in name:
            kp, kd, default = 20.0, 1.0, (-0.2 if "pitch" in name else 0.0)
        elif "hip" in name:
            kp, kd, default = 100.0, 2.5, (-0.1 if "pitch" in name else 0.0)
        elif "waist" in name:
            kp, kd, default = 100.0, 2.5, 0.0
        elif "wrist" in name:
            kp, kd, default = 10.0, 1.0, 0.0
        else:
            kp, kd, default = 40.0, 2.0, (0.3 if "elbow" in name else 0.0)
        actuator_class = ImplicitActuatorCfg if actuator_type == "implicit" else G1DelayedPDActuatorCfg
        extra = (
            {}
            if actuator_type == "implicit"
            else dict(min_delay=min_delay, max_delay=max_delay, effort_limit=effort, velocity_limit=velocity)
        )
        actuators[name] = actuator_class(
            **extra,
            joint_names_expr=[name],
            stiffness=kp,
            damping=kd,
            effort_limit_sim=effort,
            velocity_limit_sim=velocity,
        )
        scales.append(0.25 * effort / kp)
        defaults[name] = default
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            variants={"Physics": "physx"},
            activate_contact_sensors=False,
            rigid_props=PhysxRigidBodyCfg(disable_gravity=False, linear_damping=0.0, angular_damping=0.0),
            articulation_props=[PhysxArticulationCfg(enabled_self_collisions=True)],
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.76), joint_pos=defaults, joint_vel={".*": 0.0}),
        soft_joint_pos_limit_factor=0.9,
        actuators=actuators,
    ), scales

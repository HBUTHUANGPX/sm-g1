"""Compile the G1 URDF to an MJCF with a free-floating pelvis (same kinematics and inertias)."""

import argparse
from pathlib import Path
import os
import re
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data" / "g1",
                        help="Asset directory (default: repository data/g1)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import mujoco

    root = args.data_root / "g1_description"
    src = str(root / "g1_29dof_rev_1_0.urdf")
    tmp = str(root / "_g1_mj.urdf")
    out = str(root / "g1_29dof.xml")

    text = open(src).read()
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    # keep every link as its own body so the tracked link frames exist, and keep visuals for rendering
    text = re.sub(
        r"<mujoco>.*?</mujoco>",
        '<mujoco><compiler meshdir="." balanceinertia="true" discardvisual="false" fusestatic="false" strippath="false"/></mujoco>',
        text,
        flags=re.S,
    )
    open(tmp, "w").write(text)
    model = mujoco.MjModel.from_xml_path(tmp)
    mujoco.mj_saveLastXML(out, model)
    os.remove(tmp)

    xml = open(out).read()
    # free-floating pelvis
    xml = re.sub(r'(<body name="pelvis"[^>]*>)', r'\1\n      <freejoint name="floating_base_joint"/>', xml, count=1)
    if '<body name="pelvis"' not in xml:
        # URDF root link becomes worldbody children; wrap them in a pelvis body
        sys.exit("pelvis body not found; inspect " + out)
    open(out, "w").write(xml)
    m2 = mujoco.MjModel.from_xml_path(out)
    print("nq", m2.nq, "nv", m2.nv, "nbody", m2.nbody, "ngeom", m2.ngeom, "mass", m2.body_mass.sum())
    print([m2.joint(i).name for i in range(m2.njnt)])


if __name__ == "__main__":
    main()

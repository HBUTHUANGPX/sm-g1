"""Convert G1 LAFAN1 retarget CSVs to the original MotionLib NPZ schema.

Source: lvhaidong/LAFAN1_Retargeting_Dataset, pinned by the repository downloader.
CSV: root xyz metres, quaternion xyzw, 29 joint positions radians, at 30 Hz.
NPZ: 50 Hz, quaternion wxyz, world root velocities. Only the root body is stored:
the original MotionLib recomputes tracked-body quantities through its URDF FK.
"""

from pathlib import Path
import hashlib
import io
import tempfile
import numpy as np
import torch
from mimic_baseline.tasks.supermimic.robots.g1.kinematics import G1_JOINT_NAMES
from mimic_baseline.tasks.supermimic.core.rotations import quat_slerp, quat_to_mat, so3_log


def convert_clip(source, destination):
    source, destination = Path(source), Path(destination)
    source_bytes = source.read_bytes()
    raw = np.loadtxt(io.BytesIO(source_bytes), delimiter=",", dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 36 or raw.shape[0] < 2 or not np.isfinite(raw).all():
        raise ValueError("Expected finite [T>=2,36] G1 LAFAN1 CSV")
    data = torch.from_numpy(raw)
    quat = data[:, [6, 3, 4, 5]]
    norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    if (norm < 1e-8).any():
        raise ValueError("Zero root quaternion")
    quat = quat / norm
    count = int(np.floor((len(raw) - 1) / 30 / 0.02 + 1e-9)) + 1
    t = torch.arange(count, dtype=torch.float64) * 0.02 * 30
    i = t.floor().long().clamp(max=len(raw) - 2)
    alpha = (t - i).clamp(0, 1)
    pos = torch.lerp(data[i, :3], data[i + 1, :3], alpha[:, None])
    joints = torch.lerp(data[i, 7:], data[i + 1, 7:], alpha[:, None])
    q = quat_slerp(quat[i], quat[i + 1], alpha)
    R = quat_to_mat(q)
    velocity = torch.gradient(pos, spacing=(0.02,), dim=(0,))[0]
    joint_velocity = torch.gradient(joints, spacing=(0.02,), dim=(0,))[0]
    omega = torch.empty_like(pos)
    omega[0] = so3_log(R[1] @ R[0].T) / 0.02
    omega[-1] = so3_log(R[-1] @ R[-2].T) / 0.02
    omega[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / 0.04
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = lambda x: x.numpy().astype(np.float32)
    payload = dict(
        fps=np.float32(50),
        robot_name="unitree_g1",
        quaternion_order="wxyz",
        joint_names=np.asarray(G1_JOINT_NAMES),
        body_names=np.asarray(["pelvis"]),
        joint_pos=array(joints),
        joint_vel=array(joint_velocity),
        body_pos_w=array(pos[:, None]),
        body_quat_w=array(q[:, None]),
        body_lin_vel_w=array(velocity[:, None]),
        body_ang_vel_w=array(omega[:, None]),
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        schema_version=np.int32(1),
    )
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **payload)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def ensure_clip(source, destination):
    """Return the consumed source digest, refreshing stale or corrupt cache files."""
    source, destination = Path(source), Path(destination)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        with np.load(destination, allow_pickle=False) as cache:
            if (
                str(cache["source_sha256"]) == digest
                and int(cache["schema_version"]) == 1
                and str(cache["robot_name"]) == "unitree_g1"
            ):
                return digest
    except (OSError, ValueError, KeyError, EOFError):
        pass
    convert_clip(source, destination)
    with np.load(destination, allow_pickle=False) as cache:
        return str(cache["source_sha256"])

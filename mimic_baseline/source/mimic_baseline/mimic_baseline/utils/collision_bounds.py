"""Conservative collision bounds for terrain-aware recovery initialization."""

import torch


def read_collision_bounds(usd_path, *, physics_variant=None):
    """Read link-local collision boxes, including invisible USD guide shapes."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    if physics_variant is not None:
        for prim in stage.Traverse():
            variants = prim.GetVariantSet("Physics")
            if variants and physics_variant in variants.GetVariantNames():
                variants.SetVariantSelection(physics_variant)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"], False, True)
    bounds = {}
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        body = prim
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if body:
            box = cache.ComputeRelativeBound(prim, body).ComputeAlignedRange()
            name = body.GetName()
            bounds[name] = box if name not in bounds else bounds[name].UnionWith(box)
    return {name: (tuple(box.GetMin()), tuple(box.GetMax())) for name, box in bounds.items()}


def _oriented_box_intersects_aabb(corners, boxes):
    """Separating-axis test for the packed broad-phase candidate pairs."""
    half_edges = (corners[:, [4, 2, 1]] - corners[:, :1]) / 2
    axes = torch.nn.functional.normalize(half_edges, dim=-1)
    world_axes = torch.eye(3, device=corners.device, dtype=corners.dtype).expand(len(corners), -1, -1)
    cross_axes = torch.linalg.cross(axes[:, :, None], world_axes[:, None], dim=-1).flatten(1, 2)
    axes = torch.cat((axes, world_axes, cross_axes), dim=1)
    delta = boxes.mean(dim=1) - corners.mean(dim=1)
    distance = torch.einsum("nak,nk->na", axes, delta).abs()
    body_radius = torch.einsum("nak,nbk->nab", axes, half_edges).abs().sum(dim=-1)
    obstacle_radius = (axes.abs() * ((boxes[:, 1] - boxes[:, 0]) / 2)[:, None]).sum(dim=-1)
    return (distance <= body_radius + obstacle_radius + 1e-6).all(dim=-1)


def terrain_clearance_lift(corners, obstacles, floor_z, clearance):
    """Lift only intersecting spawns; clear states keep their baseline height.

    Body corners have shape [env, body, 8, xyz], ordered by local XYZ bits.
    Obstacles are world AABBs with shape [env, obstacle, min/max, xyz].
    Once an intersection is found, clear all overlying obstacles in one move.
    """
    bodies = torch.stack((corners.amin(dim=2), corners.amax(dim=2)), dim=2)
    floor_depth = floor_z - bodies[:, :, 0, 2].amin(dim=1)
    required = floor_depth
    penetrating = floor_depth > 0
    if obstacles.shape[1]:
        overlap_xy = (
            (bodies[:, :, None, 0, :2] < obstacles[:, None, :, 1, :2])
            & (bodies[:, :, None, 1, :2] > obstacles[:, None, :, 0, :2])
        ).all(dim=-1)
        depth = obstacles[:, None, :, 1, 2] - bodies[:, :, None, 0, 2]
        intersect = overlap_xy & (depth > 0) & (bodies[:, :, None, 1, 2] > obstacles[:, None, :, 0, 2])
        # World AABBs overestimate tilted links; confirm only the candidate pairs.
        env_ids, body_ids, obstacle_ids = intersect.nonzero(as_tuple=True)
        intersect[env_ids, body_ids, obstacle_ids] = _oriented_box_intersects_aabb(
            corners[env_ids, body_ids], obstacles[env_ids, obstacle_ids],
        )
        penetrating |= intersect.any(dim=(1, 2))
        required = torch.maximum(required, torch.where(overlap_xy, depth, -torch.inf).amax(dim=(1, 2)))
    return torch.where(penetrating, required.clamp_min(0) + clearance, 0)

"""USD instance traversal regression: G1 meshes are instance proxies."""
from pxr import Usd, UsdGeom, UsdPhysics
import pytest
from mimic_baseline.utils.collision_bounds import read_collision_bounds


def test_instance_proxy_collision_is_included(tmp_path):
    source = Usd.Stage.CreateNew(str(tmp_path / 'shape.usda'))
    root = UsdGeom.Xform.Define(source, '/Shape').GetPrim()
    source.SetDefaultPrim(root)
    box = UsdGeom.Cube.Define(source, '/Shape/Collision')
    box.CreateSizeAttr(2.)
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    source.GetRootLayer().Save()
    stage = Usd.Stage.CreateNew(str(tmp_path / 'robot.usda'))
    body = UsdGeom.Xform.Define(stage, '/Robot').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    instance = UsdGeom.Xform.Define(stage, '/Robot/Instance').GetPrim()
    instance.GetReferences().AddReference(str(tmp_path / 'shape.usda'))
    instance.SetInstanceable(True)
    stage.GetRootLayer().Save()
    bounds = read_collision_bounds(str(tmp_path / 'robot.usda'))
    assert bounds['Robot'][0] == pytest.approx((-1., -1., -1.))
    assert bounds['Robot'][1] == pytest.approx((1., 1., 1.))

import math
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Fuse

from cadmodel.model import CADModel


def _compute_volume(solid):
    props = GProp_GProps()
    brepgprop.VolumeProperties(solid, props)
    return abs(props.Mass())


def _fit_shape_to_unit_cube(shape, use_triangulation=True, margin=0.0, tol=1e-12):
    """
    Uniformly scale and center a TopoDS_Shape into the origin-centered [-1,1]^3 cube.
    - use_triangulation: whether to use triangulation for a tighter bounding box.
    - margin: optional inner margin (e.g. 0.02 leaves 2 percent padding each side).
    - tol: threshold to treat dimensions as zero.
    Returns: scaled_shape
    """
    bbox = Bnd_Box()
    bbox.SetGap(0.0)
    brepbndlib.Add(shape, bbox, use_triangulation)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

    if any(
        map(
            lambda v: math.isinf(v) or math.isnan(v),
            [xmin, ymin, zmin, xmax, ymax, zmax],
        )
    ):
        raise ValueError("Bounding box contains invalid numbers (inf/NaN).")

    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    cx, cy, cz = (xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5

    longest = max(dx, dy, dz)
    if longest < tol:
        # Degenerate-to-point case: only translation is needed.
        tr_translate = gp_Trsf()
        tr_translate.SetTranslation(gp_Vec(-cx, -cy, -cz))
        shape_centered = BRepBuilderAPI_Transform(shape, tr_translate, True).Shape()
        return shape_centered, 1.0, (cx, cy, cz)

    tr_translate = gp_Trsf()
    tr_translate.SetTranslation(gp_Vec(-cx, -cy, -cz))
    shape_centered = BRepBuilderAPI_Transform(shape, tr_translate, True).Shape()

    target_side = 2.0 * (1.0 - margin)
    s = target_side / longest

    tr_scale = gp_Trsf()
    tr_scale.SetScale(gp_Pnt(0.0, 0.0, 0.0), s)
    shape_scaled = BRepBuilderAPI_Transform(shape_centered, tr_scale, True).Shape()

    return shape_scaled


def intersection_over_union(pred: CADModel, gt: CADModel):
    gt_normalized = _fit_shape_to_unit_cube(
        gt.build_model().topods_shape(), use_triangulation=True
    )
    pred_normalized = _fit_shape_to_unit_cube(
        pred.build_model().topods_shape(), use_triangulation=True
    )
    intersection = BRepAlgoAPI_Common(pred_normalized, gt_normalized).Shape()
    intersection_volume = _compute_volume(intersection)

    union = BRepAlgoAPI_Fuse(
        pred.build_model().topods_shape(), gt.build_model().topods_shape()
    ).Shape()
    union_volume = _compute_volume(union)

    if union_volume == 0:
        return 0.0
    return max(0.0, min(1.0, intersection_volume / union_volume))

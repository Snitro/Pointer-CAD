import os
import torch
import trimesh
import tempfile
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from torch.distributed import get_rank
from occwl.face import Face
from occwl.edge import Edge
from occwl.solid import Solid
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.GeomAdaptor import GeomAdaptor_Curve
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Edge, topods

from cadmodel.coordinatesys import CoordinateSystem

ROUND_JSON = 6

MAX_PART_LENGTH = 64
MAX_VECTOR_LENGTH = 256

EXTRUDE_OPERATIONS = [
    "NewBodyFeatureOperation",
    "JoinFeatureOperation",
    "CutFeatureOperation",
    "IntersectFeatureOperation",
]
EXTENT_TYPE = [
    "OneSideFeatureExtentType",
    "SymmetricFeatureExtentType",
    "TwoSidesFeatureExtentType",
]

STANDARD_PLANES = {
    "Top": np.array([0, 0, 1]),
    "Right": np.array([1, 0, 0]),
    "Front": np.array([0, -1, 0]),
}

TOKEN = [
    "<|padding|>",
    "<|model_end|>",
    "<|part_end|>",
    "<|sketch_start|>",
    "<|extrude_start|>",
    "<|chamfer_start|>",
    "<|fillet_start|>",
    "<|profile_start|>",
    "<|loop_start|>",
    "<|curve_start|>",
    "<|pointer_enable|>",
    "<|pointer_disable|>",
    "<|clockwise|>",
    "<|counter_clockwise|>",
    "<|direction_x+|>",
    "<|direction_x-|>",
    "<|direction_y+|>",
    "<|direction_y-|>",
    "<|direction_z+|>",
    "<|direction_z-|>",
    "<|extrude_new|>",
    "<|extrude_join|>",
    "<|extrude_cut|>",
    "<|extrude_intersect|>",
]


def token_quantize(data, min_val=0.0, max_val=1.0, bit=8):
    if bit == 0:
        return data

    q_data = quantize(data, min_val, max_val, bit)
    return q_data + len(TOKEN)


def token_dequantize(data, min_val=0.0, max_val=1.0, bit=8):
    if bit == 0:
        return data

    if isinstance(data, (int, float)):
        return dequantize(data - len(TOKEN), min_val, max_val, bit)
    elif isinstance(data, list):
        return dequantize([d - len(TOKEN) for d in data], min_val, max_val, bit)


def quantize(data, min_val=0.0, max_val=1.0, bit=8):
    if bit == 0:
        return data

    levels = 2**bit - 1
    range_val = max_val - min_val
    if range_val <= 0:
        raise ValueError("max_val must be greater than min_val")

    def _quantize_scalar(x):
        scaled = (x - min_val) / range_val
        scaled = min(max(scaled, 0.0), 1.0)
        return int(round(scaled * levels))

    if isinstance(data, (int, float)):
        return _quantize_scalar(data)
    elif isinstance(data, np.ndarray):
        return np.round(
            np.clip((data - min_val) / range_val, 0.0, 1.0) * levels
        ).astype(np.int32)
    elif isinstance(data, list):
        return [_quantize_scalar(x) for x in data]
    elif isinstance(data, tuple):
        return tuple(_quantize_scalar(x) for x in data)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def dequantize(quantized_data, min_val=0.0, max_val=1.0, bit=8):
    if bit == 0:
        return quantized_data

    levels = 2**bit - 1
    range_val = max_val - min_val
    if range_val <= 0:
        raise ValueError("max_val must be greater than min_val")

    def _dequantize_scalar(q):
        return float(q) / levels * range_val + min_val

    if isinstance(quantized_data, (int, float)):
        return _dequantize_scalar(quantized_data)
    elif isinstance(quantized_data, np.ndarray):
        return quantized_data.astype(np.float64) / levels * range_val + min_val
    elif isinstance(quantized_data, list):
        return [_dequantize_scalar(q) for q in quantized_data]
    elif isinstance(quantized_data, tuple):
        return tuple(_dequantize_scalar(q) for q in quantized_data)
    else:
        raise TypeError(f"Unsupported data type: {type(quantized_data)}")


def quantization_mapping(value, min_val=0.0, max_val=1.0, bit=8):
    if bit == 0:
        return value

    quantized = quantize(value, min_val, max_val, bit)
    dequantized = dequantize(quantized, min_val, max_val, bit)
    return dequantized


def find_edge_by_points_and_length(
    shape: TopoDS_Compound,
    points: List[Tuple[float, float, float]],
    length: float,
    length_tolerance: float = 1e-2,
    point_tolerance: float = 1e-4,
) -> TopoDS_Edge:
    """
    Find the edge (TopoDS_Edge) that best matches a point sequence and target length.

    Candidates are ranked by length error first, then by point projection error.

    Args:
        shape: TopoDS_Compound
        points: List of points, each as (x, y, z)
        length: Expected edge length
        length_tolerance: Max allowed length deviation
        point_tolerance: Point projection distance tolerance

    Returns:
        Best matching TopoDS_Edge, or None if no valid match is found.
    """

    def points_on_edge(edge, pts, tol):
        sum_error = 0
        curve_handle, first, last = BRep_Tool.Curve(edge)
        if curve_handle is None:
            return -1
        for pt in pts:
            pnt = gp_Pnt(*pt)
            projector = GeomAPI_ProjectPointOnCurve(pnt, curve_handle)
            if projector.NbPoints() == 0:
                return -1
            dist = projector.LowerDistance()
            if dist > tol:
                return -1
            sum_error += dist
        return sum_error

    candidates = []
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        edge = topods.Edge(exp.Current())
        curve_handle, first, last = BRep_Tool.Curve(edge)
        if curve_handle is None:
            exp.Next()
            continue

        if length >= 0:
            edge_length = GCPnts_AbscissaPoint.Length(
                GeomAdaptor_Curve(curve_handle, first, last)
            )
            length_err = abs(edge_length - length)
            if length_err > length_tolerance:
                exp.Next()
                continue
        else:
            length_err = 0

        point_error = points_on_edge(edge, points, point_tolerance)
        if point_error >= 0:
            candidates.append((edge, length_err, point_error))
        exp.Next()

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[1], x[2]))
    best_edge = candidates[0][0]
    return best_edge


def project_point_to_edge(
    csys: CoordinateSystem, point: list, edge: Edge, tolerance: float = 0.2
):
    def distance(p1, p2) -> float:
        return (
            (p1.X() - p2.X()) ** 2 + (p1.Y() - p2.Y()) ** 2 + (p1.Z() - p2.Z()) ** 2
        ) ** 0.5

    point_coords = csys.relative2world(point)
    world_point = gp_Pnt(*point_coords)
    projector = GeomAPI_ProjectPointOnCurve(world_point, edge.curve())
    if projector.NbPoints() > 0:
        closest_point = projector.NearestPoint()
        if distance(world_point, closest_point) < tolerance:
            world_point = np.array(
                [closest_point.X(), closest_point.Y(), closest_point.Z()]
            )
            return csys.world2relative(world_point)[:2]
        else:
            return np.array(point)
    return None


def get_sketch_plane_normal(plane):
    if isinstance(plane, str):
        return STANDARD_PLANES[plane], np.array([0, 0, 0])
    elif isinstance(plane, Face):
        face_origin_uv = plane.uv_bounds().center()
        face_origin = plane.point(face_origin_uv)
        face_normal = plane.normal(face_origin_uv)
        face_normal = face_normal / np.linalg.norm(face_normal)
        return face_normal, face_origin
    else:
        raise ValueError("Unable to parse sketch plane normal vector.")


def match_sketch_plane_from_solid(
    solid: Solid,
    coordinate: CoordinateSystem,
    include_standard_planes: bool = False,
    normal_tol: float = 1e-4,
    point_tol: float = 1e-4,
):
    """
    Match a sketch plane against planar faces in a solid.

    Args:
        solid: Solid object.
        coordinate: Sketch plane coordinate system.
        include_standard_planes: Whether to also check XOY/YOZ/XOZ planes.
        normal_tol: Normal-vector tolerance.
        point_tol: Point-to-plane distance tolerance.

    Returns:
        List of matched Face objects.
    """
    matched_faces = []
    matched_planes = []

    sketch_origin = np.array(coordinate.origin)
    sketch_normal = np.array(coordinate.normal)
    sketch_normal = sketch_normal / np.linalg.norm(sketch_normal)

    if solid is not None:
        for face in solid.faces():
            if face.surface_type() != "plane":
                continue

            face_origin_uv = face.uv_bounds().center()
            face_origin = face.point(face_origin_uv)
            face_normal = face.normal(face_origin_uv)
            face_normal = face_normal / np.linalg.norm(face_normal)

            dot = np.dot(sketch_normal, face_normal)
            if abs(abs(dot) - 1.0) > normal_tol:
                continue

            vec = face_origin - sketch_origin
            dist = abs(np.dot(vec, sketch_normal))
            if dist > point_tol:
                continue

            matched_faces.append(face)

    if include_standard_planes:
        for name, std_n in STANDARD_PLANES.items():
            dot_std = np.dot(sketch_normal, std_n)
            if abs(abs(dot_std) - 1.0) > normal_tol:
                continue

            dist_std = abs(np.dot(std_n, sketch_origin))
            if dist_std > point_tol:
                continue
            matched_planes.append(name)

    return matched_faces, matched_planes


def match_curve_from_edges(
    edges: List[Edge],
    csys: CoordinateSystem,
    xy: Tuple[float, float],
    tolerance: float = 1e-4,
) -> List[Edge]:
    """
    Find all edges that overlap a given point within tolerance.

    Args:
    - edges: List of Edge objects.
    - csys: CoordinateSystem object with relative2world(xy).
    - xy: 2D coordinate (x, y).
    - tolerance: Matching tolerance.

    Returns:
    - List of matched edges (can be empty).
    """

    def distance(p1, p2) -> float:
        return (
            (p1.X() - p2.X()) ** 2 + (p1.Y() - p2.Y()) ** 2 + (p1.Z() - p2.Z()) ** 2
        ) ** 0.5

    point_coords = csys.relative2world(xy)
    world_point = gp_Pnt(*point_coords)
    matched_edges = []

    for edge in edges:
        curve = edge.curve()
        projector = GeomAPI_ProjectPointOnCurve(world_point, curve)
        if projector.NbPoints() > 0:
            closest_point = projector.NearestPoint()
            if distance(world_point, closest_point) <= tolerance:
                matched_edges.append(edge)

    return matched_edges


class DummyTQDM:
    def __init__(self, iterable, *args, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable if self.iterable else [])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass


def get_progress_bar(iterable=None, **kwargs):
    if (get_rank() % torch.cuda.device_count()) == 0:
        return tqdm(iterable, **kwargs)
    else:
        return DummyTQDM(iterable)


def create_mesh(
    model, mode="ascii", linear_deflection=0.001, angular_deflection=0.5
) -> trimesh.Trimesh:
    """
    Create a trimesh.Trimesh object from an occwl.Solid model.

    Parameters:
        model (Solid): The occwl Solid to be meshed.
        mode (str): 'ascii' or 'binary' STL export mode. Default is 'ascii'.
        linear_deflection (float): Controls mesh accuracy. Smaller = finer mesh.
        angular_deflection (float): Controls angular tolerance. Smaller = more accurate.

    Returns:
        trimesh.Trimesh: The mesh representation of the model.
    """
    a_shape = model.build_model().topods_shape()

    if a_shape.IsNull():
        raise ValueError("Input model shape is null.")
    if mode not in ["ascii", "binary"]:
        raise ValueError("mode must be 'ascii' or 'binary'")

    # Perform OpenCascade mesh generation
    mesh = BRepMesh_IncrementalMesh(
        a_shape, linear_deflection, False, angular_deflection, True
    )
    mesh.Perform()
    if not mesh.IsDone():
        raise RuntimeError("Mesh generation failed.")

    # Create a temporary STL file
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as temp_file:
        filename = temp_file.name

    stl_writer = StlAPI_Writer()
    stl_writer.SetASCIIMode(mode == "ascii")
    stl_writer.Write(a_shape, filename)

    if not os.path.exists(filename):
        raise IOError(f"Temporary STL file not created: {filename}")

    try:
        mesh = trimesh.load(filename, force="mesh")
    finally:
        os.remove(filename)

    return mesh

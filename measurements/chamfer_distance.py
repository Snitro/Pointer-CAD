import numpy as np

from scipy.spatial import cKDTree
from trimesh.sample import sample_surface, sample_surface_even

from misc import create_mesh
from cadmodel.model import CADModel


def _normalize(points, centered=True):
    if centered:
        centroid = np.mean(points, axis=0)
        points = points - centroid
    scale = np.abs(np.max(points) - np.min(points))
    points = points / scale
    return points


def chamfer_distance(
    pred: CADModel, gt: CADModel, points=1024, type="uniform", normalize=True
):
    pred_mesh, gt_mesh = create_mesh(pred), create_mesh(gt)
    pred_points, gt_points = None, None

    if type == "uniform":
        pred_points, _ = sample_surface(pred_mesh, points)
        gt_points, _ = sample_surface(gt_mesh, points)
    elif type == "even":
        pred_points, _ = sample_surface_even(pred_mesh, points)
        gt_points, _ = sample_surface_even(gt_mesh, points)
    else:
        raise AssertionError(f"Unknown sample type {type}")

    if normalize:
        pred_points = _normalize(pred_points)
        gt_points = _normalize(gt_points)

    # one direction
    pred_points_kd_tree = cKDTree(pred_points)
    one_distances, one_vertex_ids = pred_points_kd_tree.query(gt_points)
    gt_to_pred_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = cKDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_points)
    pred_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_pred_chamfer + pred_to_gt_chamfer

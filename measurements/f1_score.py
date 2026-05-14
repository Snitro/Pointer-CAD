import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score as sklearn_f1_score

from occwl.solid import Solid
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh

from cadmodel.profile import Loop, Curve
from cadmodel.model import CADModel
from cadmodel.extrude import Extrude
from cadmodel.sketch import Sketch


def point_distance(p1, p2, type="l2"):
    if type.lower() == "l2":
        return np.sqrt(np.sum((p1 - p2) ** 2))
    elif type.lower() == "l1":
        return np.sum(np.abs(p1 - p2))
    else:
        raise NotImplementedError(f"Distance type {type} not yet supported")


def extract_curves(loop: Loop):
    if loop is None:
        return []

    curves = []
    for curve in loop.curves:
        points = curve.sample()
        bbox = np.array([np.min(points, axis=0), np.max(points, axis=0)])
        curves.append((curve, bbox))

    return curves


def extract_loops(sketch: Sketch):
    if sketch is None:
        return []

    loops = []
    for profile_name, profile in sketch.profile_data.items():
        for loop in profile.loops:
            points_list = []
            for curve in loop.curves:
                points = curve.sample()
                points_3d = np.hstack([points, np.zeros((points.shape[0], 1))])
                coordinate = sketch.sketch_data[sketch.profile2sketch[profile_name]]
                world_positions = coordinate.relative2world(points_3d)
                ref_positions = sketch.ref_coordinate.world2relative(world_positions)
                ref_positions[np.abs(ref_positions[:, 2]) < 1e-6, 2] = 0
                points_list.append(ref_positions)
            points = np.vstack(points_list)
            bbox = np.array([np.min(points, axis=0), np.max(points, axis=0)])
            if np.any(np.isnan(bbox)):
                raise ValueError(f"{bbox}")
            loops.append((loop, bbox))

    return loops


def create_matched_pair(list1, list2, row_indices, col_indices):
    """
    Creates a list of matched pairs based on the row and column indices.

    Args:
        list1 (list): The first list of elements.
        list2 (list): The second list of elements.
        row_indices (list): List of row indices based on Hungarian Matching
        col_indices (list): List of row indices based on Hungarian Matching

    Returns:
        list: List of matched pairs, where each pair is a list containing an element from list1 and an element from list2.
    """
    assert len(row_indices) == len(col_indices)

    def get_element(item):
        return item[0] if item is not None else None

    matched_pair = []
    for idx_1, idx_2 in zip(row_indices, col_indices):
        list1_element = list1[idx_1] if idx_1 < len(list1) else None
        list2_element = list2[idx_2] if idx_2 < len(list2) else None
        matched_pair.append([get_element(list1_element), get_element(list2_element)])

    return matched_pair


def loop_match(gt_curve: Loop, pred_curve: Loop, scale=1, multiplier=1):
    gt_curves = extract_curves(gt_curve)
    pred_cureves = extract_curves(pred_curve)

    n_max = max(len(gt_curves), len(pred_cureves))
    cost_matrix = np.ones((n_max, n_max)) * multiplier

    for idx_pred in range(len(pred_cureves)):
        for idx_gt in range(len(gt_curves)):
            pred_curve, pred_bbox = pred_cureves[idx_pred]
            gt_curve, gt_bbox = gt_curves[idx_gt]

            if pred_curve is None or gt_curve is None:
                continue

            cost_matrix[idx_gt, idx_pred] = point_distance(
                pred_bbox * scale, gt_bbox * scale
            )

    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matched_curve_pair = create_matched_pair(
        gt_curves, pred_cureves, row_indices, col_indices
    )

    return matched_curve_pair


def sketch_match(gt_sketch: Sketch, pred_sketch: Sketch, scale=1, multiplier=1):
    gt_loops = extract_loops(gt_sketch)
    pred_loops = extract_loops(pred_sketch)

    n_max = max(len(pred_loops), len(gt_loops))
    cost_matrix = np.ones((n_max, n_max)) * multiplier

    for idx_pred in range(len(pred_loops)):
        for idx_gt in range(len(gt_loops)):
            pred_loop, pred_bbox = pred_loops[idx_pred]
            gt_loop, gt_bbox = gt_loops[idx_gt]

            if pred_loop is None or gt_loop is None:
                continue

            cost_matrix[idx_gt, idx_pred] = point_distance(
                pred_bbox * scale, gt_bbox * scale
            )

    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matched_loop_pair = create_matched_pair(
        gt_loops, pred_loops, row_indices, col_indices
    )

    matched_curve_pair = []
    for pair in matched_loop_pair:
        matched_curve_pair += loop_match(pair[0], pair[1], scale, multiplier)

    return matched_curve_pair


def _extract_sketch(model: CADModel):
    extrudes: list[Extrude] = [
        feature for feature in model.seq if isinstance(feature, Extrude)
    ]
    sketches: list[Sketch] = [
        extrude.sketches for extrude in extrudes if extrude.sketches is not None
    ]

    return sketches


def f1_score(pred: CADModel, gt: CADModel):
    pred_sketches = _extract_sketch(pred)
    gt_sketches = _extract_sketch(gt)

    n_max = max(len(pred_sketches), len(gt_sketches))

    if len(pred_sketches) < n_max:
        pred_sketches += [None] * (n_max - len(pred_sketches))
    elif len(gt_sketches) < n_max:
        gt_sketches += [None] * (n_max - len(gt_sketches))

    matched_curve_pair: list[list[Curve, Curve]] = []
    for i in range(n_max):
        matched_curve_pair += sketch_match(gt_sketches[i], pred_sketches[i])

    gt_curve_list = [
        curve.curve_type if curve is not None else "none"
        for curve, _ in matched_curve_pair
    ]
    pred_curve_list = [
        curve.curve_type if curve is not None else "none"
        for _, curve in matched_curve_pair
    ]
    unique_curve_labels = np.union1d(
        np.unique(gt_curve_list), np.unique(pred_curve_list)
    ).tolist()

    curve_f1 = sklearn_f1_score(
        gt_curve_list, pred_curve_list, labels=unique_curve_labels, average=None
    )

    result = {}
    for label, score in zip(unique_curve_labels, curve_f1):
        if label == "none":
            continue
        result[label] = score

    gt_operation_list = [operation.__class__.__name__.lower() for operation in gt.seq]
    pred_operation_list = [
        operation.__class__.__name__.lower() for operation in pred.seq
    ]

    n_max = max(len(gt_operation_list), len(pred_operation_list))
    if len(gt_operation_list) < n_max:
        gt_operation_list += ["none"] * (n_max - len(gt_operation_list))
    elif len(pred_operation_list) < n_max:
        pred_operation_list += ["none"] * (n_max - len(pred_operation_list))
    unique_operation_labels = np.union1d(
        np.unique(gt_operation_list), np.unique(pred_operation_list)
    ).tolist()

    operation_f1 = sklearn_f1_score(
        gt_operation_list,
        pred_operation_list,
        labels=unique_operation_labels,
        average=None,
    )

    for label, score in zip(unique_operation_labels, operation_f1):
        if label == "none":
            continue
        result[label] = score

    return result

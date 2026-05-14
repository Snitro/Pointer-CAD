import cv2
import math
import numpy as np
from functools import partial
from occwl.edge import Edge
from occwl.wire import Wire
from collections import OrderedDict

from .coordinatesys import CoordinateSystem
from misc import (
    match_sketch_plane_from_solid,
    match_curve_from_edges,
    get_sketch_plane_normal,
    project_point_to_edge,
    TOKEN,
)
from .profile import Profile, Curve, Line, Circle, Arc


class Sketch(object):

    def __init__(
        self,
        profile_data: dict[str, Profile],
        sketch_data: dict[str, CoordinateSystem],
        profile2sketch,
        sketch_name={},
    ):
        self.profile_data = profile_data
        self.sketch_data = sketch_data
        self.profile2sketch = profile2sketch
        self.sketch_name = sketch_name

    @property
    def ref_coordinate(self):
        return next(iter(self.sketch_data.values()))

    @property
    def dimension(self):
        extra_height = 0
        scale = 1

        if len(self.sketch_data) == 1:
            scale = self.ref_coordinate.scale

            curve_num = 0
            circle_num = 0
            circle_radius = 0

            global_points_list = []
            for profile_name, profile in self.profile_data.items():
                for loop in profile.loops:
                    for curve in loop.curves:
                        curve_num += 1
                        if isinstance(curve, Circle):
                            circle_num += 1
                            circle_radius = curve.radius

                        points = curve.sample()
                        global_points_list.append(points)

            if curve_num == 1 and circle_num == 1:
                return (
                    circle_radius * 2 * scale,
                    circle_radius * 2 * scale,
                    extra_height * scale,
                )

            points_2d = np.vstack(global_points_list).astype(np.float32)
        else:
            ref_coordinate = self.ref_coordinate
            scale = ref_coordinate.scale

            global_points_list = []
            for profile_name, profile in self.profile_data.items():
                for loop in profile.loops:
                    for curve in loop.curves:
                        points = curve.sample()
                        points_3d = np.hstack([points, np.zeros((points.shape[0], 1))])
                        coordinate = self.sketch_data[self.profile2sketch[profile_name]]
                        world_positions = coordinate.relative2world(points_3d)
                        ref_positions = ref_coordinate.world2relative(world_positions)
                        ref_positions[np.abs(ref_positions[:, 2]) < 1e-6, 2] = 0
                        global_points_list.append(ref_positions)
            global_points = np.vstack(global_points_list)

            extra_height = np.max(global_points[:, 2]) - np.min(global_points[:, 2])

            projected_2d = global_points[:, :2]
            points_2d = projected_2d.astype(np.float32)

        rect = cv2.minAreaRect(points_2d)
        _, (width, height), _ = rect
        dimensions = sorted([width, height], reverse=True)

        return dimensions[0] * scale, dimensions[1] * scale, abs(extra_height) * scale

    @staticmethod
    def start_token():
        return [TOKEN.index("<|sketch_start|>"), None]

    @staticmethod
    def from_dict(all_stat, sketches_ids):
        profiles = {profile: sketch for sketch, profile in sketches_ids}

        sketch_data = {
            sketch: CoordinateSystem.from_dict(
                all_stat["features"][sketch]["transform"]
            )
            for sketch in OrderedDict.fromkeys(profiles.values())
        }
        sketch_name_dict = {
            sketch: all_stat["features"][sketch]["name"]
            for sketch in sketch_data.keys()
        }

        profile_data = {}
        for profile_name, sketch_name in profiles.items():
            sketch_entity = all_stat["features"][sketch_name]
            assert sketch_entity["type"] == "Sketch"
            profile_data[profile_name] = Profile.from_dict(
                sketch_entity["profiles"][profile_name]
            )

        return Sketch(profile_data, sketch_data, profiles, sketch_name_dict)

    def transform(self, movement=(0, 0, 0), scale=1):
        for sketch, coordinate in self.sketch_data.items():
            coordinate.transform(movement=movement, scale=scale)
        for profile_name, profile in self.profile_data.items():
            for loop in profile.loops:
                for curve in loop.curves:
                    curve.transform(scale=scale)

    def build_wire(self):
        wires = {}

        for profile_name, profile in self.profile_data.items():
            sketch_name = self.profile2sketch[profile_name]
            wires[f"{sketch_name}_{profile_name}"] = []
            coordinate = self.sketch_data[sketch_name]
            for loop in profile.loops:
                edges = []
                for curve in loop.curves:
                    edge = self._build_edge(curve, coordinate)
                    if edge is not None:
                        edges.append(edge)

                if len(edges):
                    wire = Wire.make_from_edges(edges)
                    if wire is not None and wire.valid():
                        wires[f"{sketch_name}_{profile_name}"].append(wire)

        return wires

    def _build_edge(self, curve: Curve, coordinate: CoordinateSystem):
        """Build a 3D edge from a 2D sketch curve under the given coordinate system."""
        if isinstance(curve, Line):
            if np.allclose(curve.start_point, curve.end_point):
                return None
            start_point = coordinate.relative2world(curve.start_point)
            end_point = coordinate.relative2world(curve.end_point)
            edge = Edge.make_line_from_points(start_point, end_point)
        elif isinstance(curve, Circle):
            if curve.radius <= 0:
                return None
            center = coordinate.relative2world(curve.center)
            normal = coordinate.normal
            radius = abs(float(curve.radius * coordinate.scale))
            edge = Edge.make_circle(center, radius, normal)
        elif isinstance(curve, Arc):
            if np.allclose(curve.start_point, curve.end_point):
                return None
            start_point = coordinate.relative2world(curve.start_point)
            mid_point = coordinate.relative2world(curve.mid_point)
            end_point = coordinate.relative2world(curve.end_point)
            edge = Edge.make_arc_of_circle(start_point, mid_point, end_point)
        else:
            raise NotImplementedError(type(curve))

        return edge

    def _move_to_origin(self):
        for sketch, coordinate in self.sketch_data.items():
            points_list = []
            for profile_name, sketch_name in self.profile2sketch.items():
                if sketch_name == sketch:
                    profile = self.profile_data[profile_name]
                    for loop in profile.loops:
                        for curve in loop.curves:
                            points = curve.sample()
                            points_list.append(points)
            sketch_points = np.vstack(points_list)
            basepoint = np.min(sketch_points, axis=0)

            basepoint_world = coordinate.relative2world(np.append(basepoint, 0))
            coordinate.transform(basepoint_world - coordinate.origin)

            for profile_name, sketch_name in self.profile2sketch.items():
                if sketch_name == sketch:
                    profile = self.profile_data[profile_name]
                    for loop in profile.loops:
                        for curve in loop.curves:
                            curve.transform(movement=-basepoint)

    def normalize(self):
        self._move_to_origin()

        for sketch, coordinate in self.sketch_data.items():
            points_list = []
            for profile_name, sketch_name in self.profile2sketch.items():
                if sketch_name == sketch:
                    profile = self.profile_data[profile_name]
                    for loop in profile.loops:
                        for curve in loop.curves:
                            points = curve.sample()
                            points_list.append(points)
            sketch_points = np.vstack(points_list)
            bbox_min = np.min(sketch_points, axis=0)
            bbox_max = np.max(sketch_points, axis=0)

            max_dis = np.max(np.abs(bbox_max - bbox_min))
            scale = 1 / max_dis
            coordinate.scale *= float(max_dis)

            for profile_name, sketch_name in self.profile2sketch.items():
                if sketch_name == sketch:
                    profile = self.profile_data[profile_name]
                    for loop in profile.loops:
                        for curve in loop.curves:
                            curve.transform(scale=scale)

    def to_vector(self, built_solid=None, quant_bits=0):
        vec = []
        for sketch, coordinate in self.sketch_data.items():
            vec.append(self.start_token())

            matched_faces, matched_planes = match_sketch_plane_from_solid(
                built_solid, coordinate, True
            )
            operation_planes = matched_faces + matched_planes
            if len(operation_planes) == 0:
                raise RuntimeError("No sketch operation plane found")
            vec.append([TOKEN.index("<|pointer_enable|>"), operation_planes])

            vec.append(coordinate.to_vector(quant_bits))

            matched_edges = []
            for face in matched_faces:
                matched_edges.extend(list(face.edges()))

            match_curve_func = partial(
                match_curve_from_edges, matched_edges, coordinate
            )
            for profile_name, sketch_name in self.profile2sketch.items():
                if sketch_name == sketch:
                    profile = self.profile_data[profile_name]
                    vec.extend(profile.to_vector(match_curve_func, quant_bits))

        return vec

    @staticmethod
    def from_vector(vector, quant_bits=0, strict=True):
        profile_data: dict[str, Profile] = {}
        sketch_data: dict[str, CoordinateSystem] = {}
        profile2sketch: dict[str, str] = {}

        SK_positions = list(
            reversed([i for i, x in enumerate(vector) if x == Sketch.start_token()])
        )
        if len(SK_positions) == 0:
            raise ValueError("Not enough elements to decode Sketch from vector.")

        sketch_end = None
        for sketch_start in SK_positions:
            try:
                sketch_vector = vector[sketch_start + 1 : sketch_end]
                P_positions = [
                    i
                    for i, x in enumerate(sketch_vector)
                    if x[0] == TOKEN.index("<|pointer_enable|>")
                ]
                if len(P_positions) == 0:
                    raise ValueError(
                        "Not enough elements to decode Sketch from vector."
                    )
                for p_idx in P_positions:
                    try:
                        sketch_content_vector = sketch_vector[p_idx + 1 :]
                        operation_plane = sketch_vector[p_idx][1][0]
                        operation_plane_normal, operation_plane_origin = (
                            get_sketch_plane_normal(operation_plane)
                        )
                        coordinate = CoordinateSystem.from_vector(
                            sketch_content_vector,
                            operation_plane_normal,
                            operation_plane_origin,
                            quant_bits,
                        )

                        profiles = []
                        project_point_func = partial(project_point_to_edge, coordinate)
                        PF_positions = list(
                            reversed(
                                [
                                    i
                                    for i, x in enumerate(sketch_content_vector)
                                    if x == Profile.start_token()
                                ]
                            )
                        )
                        profile_end = None
                        for profile_start in PF_positions:
                            try:
                                profile_vector = sketch_content_vector[
                                    profile_start + 1 : profile_end
                                ]
                                profiles.append(
                                    Profile.from_vector(
                                        profile_vector,
                                        project_point_func,
                                        quant_bits,
                                        strict,
                                    )
                                )
                                profile_end = profile_start
                            except Exception as e:
                                if strict:
                                    raise e

                        if len(profiles) == 0:
                            raise ValueError(
                                "Not enough elements to decode Sketch from vector."
                            )

                        break
                    except Exception as e:
                        raise e
                else:
                    raise ValueError(
                        "Not enough elements to decode Sketch from vector."
                    )

                sketch_name = f"build_from_vector_{len(sketch_data)}"
                sketch_data[sketch_name] = coordinate
                for profile in profiles:
                    profile_name = f"build_from_vector_{len(profile_data)}"
                    profile_data[profile_name] = profile
                    profile2sketch[profile_name] = sketch_name

                sketch_end = sketch_start
            except Exception as e:
                if strict:
                    raise e

        sketch_data = dict(reversed(list(sketch_data.items())))

        return Sketch(profile_data, sketch_data, profile2sketch)

    def _json(self):
        sketch_json = {}

        i = 0
        for sketch, data in self.sketch_data.items():
            i += 1
            j = 1

            sketch_json[f"sketch_{i}"] = (
                {"name": self.sketch_name[sketch]} if sketch in self.sketch_name else {}
            )
            for profile_name, sketch_name in self.profile2sketch.items():
                if sketch_name == sketch:
                    sketch_json[f"sketch_{i}"][f"profile_{j}"] = self.profile_data[
                        profile_name
                    ]._json()
                    j += 1

            sketch_json[f"sketch_{i}"]["coordinate_system"] = data._json()

        return sketch_json

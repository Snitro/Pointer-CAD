import math
import numpy as np
from occwl.edge import Edge
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRep import BRep_Tool
from OCC.Core.Geom import Geom_Circle, Geom_Ellipse, Geom_TrimmedCurve

from misc import ROUND_JSON


class Curve3D:
    @staticmethod
    def from_dict(stat):
        if stat["type"] == "Line3D":
            return Line3D.from_dict(stat)
        elif stat["type"] == "Circle3D":
            return Circle3D.from_dict(stat)
        elif stat["type"] == "Arc3D":
            return Arc3D.from_dict(stat)
        elif stat["type"] == "EllipseArc3D":
            return EllipseArc3D.from_dict(stat)
        elif stat["type"] == "ComplexCurve3D":
            return ComplexCurve3D.from_dict(stat)
        else:
            raise NotImplementedError(
                "curve type not supported yet: {}".format(stat["type"])
            )

    @staticmethod
    def from_occ(edge: Edge):
        ctype = edge.curve_type()

        if ctype == "line":
            return Line3D.from_occ(edge)
        elif ctype == "circle" and edge.closed_edge():
            return Circle3D.from_occ(edge)
        elif ctype == "circle":
            return Arc3D.from_occ(edge)
        elif ctype == "ellipse":
            return EllipseArc3D.from_occ(edge)
        else:
            return ComplexCurve3D.from_occ(edge)

    @property
    def length(self):
        pass

    def sample(self, num_points=5):
        pass

    def transform(self, movement=(0, 0, 0), scale=1):
        pass

    def _json(self):
        pass


class Line3D(Curve3D):
    def __init__(self, start_point, end_point):
        super().__init__()
        self.start_point = start_point
        self.end_point = end_point

    @property
    def length(self):
        return np.linalg.norm(self.end_point - self.start_point)

    @staticmethod
    def from_dict(stat):
        assert stat["type"] == "Line3D"
        start_point = np.array(
            [
                stat["start_point"]["x"],
                stat["start_point"]["y"],
                stat["start_point"]["z"],
            ]
        )
        end_point = np.array(
            [stat["end_point"]["x"], stat["end_point"]["y"], stat["end_point"]["z"]]
        )
        return Line3D(start_point, end_point)

    @staticmethod
    def from_occ(edge: Edge):
        occ_edge = edge.topods_shape()
        curve_handle, first_param, last_param = BRep_Tool.Curve(occ_edge)

        start_point = gp_Pnt()
        curve_handle.D0(first_param, start_point)

        end_point = gp_Pnt()
        curve_handle.D0(last_param, end_point)

        start_point = np.array([start_point.X(), start_point.Y(), start_point.Z()])
        end_point = np.array([end_point.X(), end_point.Y(), end_point.Z()])

        return Line3D(start_point, end_point)

    def sample(self, num_points=5):
        num_points = max(num_points, 2)

        start = np.array(self.start_point, dtype=np.float64)
        end = np.array(self.end_point, dtype=np.float64)
        t = np.linspace(0, 1, num_points)
        points = (1 - t)[:, None] * start + t[:, None] * end
        return points.tolist()

    def transform(self, movement=(0, 0, 0), scale=1):
        self.start_point = (self.start_point + movement) * scale
        self.end_point = (self.end_point + movement) * scale

    def _json(self):
        line_json = {
            "type": "3D Line",
            "start": [round(v, ROUND_JSON) for v in self.start_point],
            "end": [round(v, ROUND_JSON) for v in self.end_point],
        }

        return line_json


class Circle3D(Curve3D):
    def __init__(self, center, radius, normal):
        super().__init__()
        self.center = center
        self.radius = radius
        self.normal = normal

    @property
    def length(self):
        return 2 * np.pi * self.radius

    @staticmethod
    def from_dict(stat):
        assert stat["type"] == "Circle3D"
        center = np.array(
            [
                stat["center_point"]["x"],
                stat["center_point"]["y"],
                stat["center_point"]["z"],
            ]
        )
        radius = stat["radius"]
        normal = np.array(
            [stat["normal"]["x"], stat["normal"]["y"], stat["normal"]["z"]]
        )
        return Circle3D(center, radius, normal)

    @staticmethod
    def from_occ(edge: Edge):
        occ_edge = edge.topods_shape()
        curve_handle, first_param, last_param = BRep_Tool.Curve(occ_edge)
        circle: Geom_Circle = Geom_Circle.DownCast(curve_handle)

        center = circle.Location()
        radius = circle.Radius()
        normal = circle.Position().Direction()

        center = np.array([center.X(), center.Y(), center.Z()])
        normal = np.array([normal.X(), normal.Y(), normal.Z()])
        return Circle3D(center, radius, normal)

    def sample(self, num_points=5):
        num_points = max(num_points, 1)

        center = np.array(self.center, dtype=np.float64)
        normal = np.array(self.normal, dtype=np.float64)
        radius = float(self.radius)

        normal = normal / np.linalg.norm(normal)
        if np.allclose(normal, [0, 0, 1]) or np.allclose(normal, [0, 0, -1]):
            u = np.array([1, 0, 0], dtype=np.float64)
        else:
            u = np.cross(normal, [0, 0, 1])
            u = u / np.linalg.norm(u)
        v = np.cross(normal, u)

        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        points = np.array(
            [
                center + radius * (np.cos(theta) * u + np.sin(theta) * v)
                for theta in angles
            ],
            dtype=np.float64,
        )

        return points.tolist()

    def transform(self, movement=(0, 0, 0), scale=1):
        self.center = (self.center + movement) * scale
        self.radius *= scale

    def _json(self):
        circle_point = self.sample(1)[0]
        circle_json = {
            "type": "3D Circle",
            "center": [round(v, ROUND_JSON) for v in self.center],
            "via": [round(v, ROUND_JSON) for v in circle_point],
        }
        return circle_json


class Arc3D(Curve3D):
    def __init__(
        self,
        start_point,
        end_point,
        center,
        mid_point,
        radius,
        sweep_angle,
        normal,
        is_counter_clockwise,
    ):
        super().__init__()
        self.start_point = start_point
        self.end_point = end_point
        self.center = center
        self.mid_point = mid_point
        self.radius = radius
        self.sweep_angle = sweep_angle
        self.normal = normal
        self.is_counter_clockwise = is_counter_clockwise

    @property
    def length(self):
        return math.radians(self.sweep_angle) * self.radius

    @staticmethod
    def from_dict(stat):
        assert stat["type"] == "Arc3D"
        start_point = np.array(
            [
                stat["start_point"]["x"],
                stat["start_point"]["y"],
                stat["start_point"]["z"],
            ]
        )
        end_point = np.array(
            [stat["end_point"]["x"], stat["end_point"]["y"], stat["end_point"]["z"]]
        )
        center = np.array(
            [
                stat["center_point"]["x"],
                stat["center_point"]["y"],
                stat["center_point"]["z"],
            ]
        )
        mid_point = np.array(
            [stat["mid_point"]["x"], stat["mid_point"]["y"], stat["mid_point"]["z"]]
        )
        radius = stat["radius"]
        sweep_angle = math.degrees(stat["end_angle"])
        normal = np.array(
            [stat["normal"]["x"], stat["normal"]["y"], stat["normal"]["z"]]
        )
        ref_vec = np.array(
            [
                stat["reference_vector"]["x"],
                stat["reference_vector"]["y"],
                stat["reference_vector"]["z"],
            ]
        )

        vec_start = start_point - center
        vec_end = end_point - center
        vec_start_unit = vec_start / np.linalg.norm(vec_start)
        vec_end_unit = vec_end / np.linalg.norm(vec_end)
        ref_vec_unit = ref_vec / np.linalg.norm(ref_vec)

        def angle_between(v1, v2):
            dot = np.dot(v1, v2)
            dot = np.clip(dot, -1.0, 1.0)
            return np.arccos(dot)

        angle_with_start = angle_between(ref_vec_unit, vec_start_unit)
        angle_with_end = angle_between(ref_vec_unit, vec_end_unit)
        tolerance = math.radians(5)

        if angle_with_start < angle_with_end:
            if angle_with_start > tolerance:
                raise ValueError("Angle between ref_vec and start_vec is too large and inconsistent")
            is_counter_clockwise = True
        else:
            if angle_with_end > tolerance:
                raise ValueError("Angle between ref_vec and end_vec is too large and inconsistent")
            is_counter_clockwise = False

        return Arc3D(
            start_point,
            end_point,
            center,
            mid_point,
            radius,
            sweep_angle,
            normal,
            is_counter_clockwise,
        )

    @staticmethod
    def from_occ(edge: Edge):
        occ_edge = edge.topods_shape()
        curve_handle, first_param, last_param = BRep_Tool.Curve(occ_edge)
        circle: Geom_Circle = Geom_Circle.DownCast(curve_handle)

        center_gp = circle.Location()
        radius = circle.Radius()
        normal = circle.Position().Direction()

        start_gp = gp_Pnt()
        end_gp = gp_Pnt()
        mid_gp = gp_Pnt()

        curve_handle.D0(first_param, start_gp)
        curve_handle.D0(last_param, end_gp)
        curve_handle.D0((first_param + last_param) / 2.0, mid_gp)

        def gp_pnt_to_np(gp_pnt):
            return np.array([gp_pnt.X(), gp_pnt.Y(), gp_pnt.Z()])

        start_point = gp_pnt_to_np(start_gp)
        end_point = gp_pnt_to_np(end_gp)
        mid_point = gp_pnt_to_np(mid_gp)
        center = gp_pnt_to_np(center_gp)
        normal = gp_pnt_to_np(normal)

        v1 = start_point - center
        v2 = end_point - center

        v1_unit = v1 / np.linalg.norm(v1)
        v2_unit = v2 / np.linalg.norm(v2)

        dot = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
        sweep_angle = np.arccos(dot)

        cross = np.cross(v1_unit, v2_unit)
        is_counter_clockwise = np.dot(cross, normal) > 0

        if not is_counter_clockwise:
            sweep_angle = 2 * np.pi - sweep_angle
        sweep_angle = math.degrees(sweep_angle)

        return Arc3D(
            start_point,
            end_point,
            center,
            mid_point,
            radius,
            sweep_angle,
            normal,
            is_counter_clockwise,
        )

    def sample(self, num_points=5):
        num_points = max(num_points, 2)

        center = np.array(self.center, dtype=np.float64)
        normal = np.array(self.normal, dtype=np.float64)
        radius = float(self.radius)

        normal = normal / np.linalg.norm(normal)
        u = (
            self.start_point if self.is_counter_clockwise else self.end_point
        ) - self.center
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)

        angles = np.linspace(0, math.radians(self.sweep_angle), num_points)

        points = np.array(
            [
                center + radius * (np.cos(theta) * u + np.sin(theta) * v)
                for theta in angles
            ],
            dtype=np.float64,
        )

        return points.tolist()

    def transform(self, movement=(0, 0, 0), scale=1):
        self.start_point = (self.start_point + movement) * scale
        self.end_point = (self.end_point + movement) * scale
        self.center = (self.center + movement) * scale
        self.mid_point = (self.mid_point + movement) * scale
        self.radius *= scale

    def _json(self):
        arc_json = {
            "type": "3D Arc",
            "start": [round(v, ROUND_JSON) for v in self.start_point],
            "end": [round(v, ROUND_JSON) for v in self.end_point],
            "via": [round(v, ROUND_JSON) for v in self.mid_point],
        }

        return arc_json


class EllipseArc3D(Curve3D):
    def __init__(
        self,
        start_point,
        end_point,
        center,
        mid_point,
        major_radius,
        minor_radius,
        normal,
        start_angle,
        end_angle,
        ref_vec,
        is_counter_clockwise,
    ):
        super().__init__()
        self.start_point = start_point
        self.end_point = end_point
        self.center = center
        self.mid_point = mid_point
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.normal = normal
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.ref_vec = ref_vec
        self.is_counter_clockwise = is_counter_clockwise

        self.start_angle_ec = self.central_to_eccentric_angle(self.start_angle)
        self.end_angle_ec = self.central_to_eccentric_angle(self.end_angle)

    @property
    def length(self):
        return -1

    def central_to_eccentric_angle(self, theta):
        """
        Convert ellipse central angle theta to eccentric angle (parameter angle) phi.
        This avoids tan(theta) divergence around angles like 90 and 270 degrees.

        Args:
            theta: float or np.ndarray, central angle.

        Returns:
            phi: float or np.ndarray, eccentric angle in the same unit as input.
        """
        theta = np.deg2rad(theta)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        phi = np.arctan2(self.minor_radius * sin_theta, self.major_radius * cos_theta)

        phi = np.mod(phi, 2 * np.pi)

        return np.rad2deg(phi)

    @staticmethod
    def from_dict(stat):
        assert stat["type"] == "EllipseArc3D"
        start_point = np.array(
            [
                stat["start_point"]["x"],
                stat["start_point"]["y"],
                stat["start_point"]["z"],
            ]
        )
        end_point = np.array(
            [stat["end_point"]["x"], stat["end_point"]["y"], stat["end_point"]["z"]]
        )
        center = np.array(
            [
                stat["center_point"]["x"],
                stat["center_point"]["y"],
                stat["center_point"]["z"],
            ]
        )
        mid_point = np.array(
            [stat["mid_point"]["x"], stat["mid_point"]["y"], stat["mid_point"]["z"]]
        )
        major_radius = stat["major_radius"]
        minor_radius = stat["minor_radius"]
        normal = np.array(
            [stat["normal"]["x"], stat["normal"]["y"], stat["normal"]["z"]]
        )
        start_vecor = np.array(
            [
                stat["start_vecor"]["x"],
                stat["start_vecor"]["y"],
                stat["start_vecor"]["z"],
            ]
        )
        end_vector = np.array(
            [stat["end_vector"]["x"], stat["end_vector"]["y"], stat["end_vector"]["z"]]
        )
        start_angle = math.degrees(stat["start_angle"])
        end_angle = math.degrees(stat["end_angle"])
        ref_vec = np.array(
            [
                stat["reference_vector"]["x"],
                stat["reference_vector"]["y"],
                stat["reference_vector"]["z"],
            ]
        )

        vec_start = start_point - center
        vec_end = end_point - center
        vec_start_unit = vec_start / np.linalg.norm(vec_start)
        vec_end_unit = vec_end / np.linalg.norm(vec_end)
        ref_vec_unit = start_vecor / np.linalg.norm(start_vecor)

        def angle_between(v1, v2):
            dot = np.dot(v1, v2)
            dot = np.clip(dot, -1.0, 1.0)
            return np.arccos(dot)

        angle_with_start = angle_between(ref_vec_unit, vec_start_unit)
        angle_with_end = angle_between(ref_vec_unit, vec_end_unit)
        tolerance = math.radians(5)

        if angle_with_start < angle_with_end:
            if angle_with_start > tolerance:
                raise ValueError("Angle between ref_vec and start_vec is too large and inconsistent")
            is_counter_clockwise = True
        else:
            if angle_with_end > tolerance:
                raise ValueError("Angle between ref_vec and end_vec is too large and inconsistent")
            is_counter_clockwise = False

        return EllipseArc3D(
            start_point,
            end_point,
            center,
            mid_point,
            major_radius,
            minor_radius,
            normal,
            start_angle,
            end_angle,
            ref_vec,
            is_counter_clockwise,
        )

    @staticmethod
    def from_occ(edge: Edge):
        occ_edge = edge.topods_shape()
        curve_handle, first_param, last_param = BRep_Tool.Curve(occ_edge)
        trimmed: Geom_TrimmedCurve = Geom_TrimmedCurve.DownCast(curve_handle)
        ellipse: Geom_Ellipse = Geom_Ellipse.DownCast(trimmed.BasisCurve())

        axis = ellipse.Position()
        center_pnt = ellipse.Location()
        major_radius = ellipse.MajorRadius()
        minor_radius = ellipse.MinorRadius()
        normal = axis.Direction()
        ref_vec = axis.XDirection()

        start_gp = gp_Pnt()
        end_gp = gp_Pnt()
        mid_gp = gp_Pnt()

        curve_handle.D0(first_param, start_gp)
        curve_handle.D0(last_param, end_gp)
        curve_handle.D0((first_param + last_param) / 2.0, mid_gp)

        def gp_to_np(gp_pnt):
            return np.array([gp_pnt.X(), gp_pnt.Y(), gp_pnt.Z()])

        start_point = gp_to_np(start_gp)
        end_point = gp_to_np(end_gp)
        mid_point = gp_to_np(mid_gp)
        center = gp_to_np(center_pnt)
        normal = gp_to_np(normal)
        ref_vec = gp_to_np(ref_vec)

        v1 = start_point - center
        v2 = end_point - center

        def angle_from_ref(vec):
            vec_norm = vec / np.linalg.norm(vec)
            dot = np.clip(np.dot(ref_vec, vec_norm), -1.0, 1.0)
            angle = np.arccos(dot)
            cross = np.cross(ref_vec, vec_norm)
            if np.dot(cross, normal) < 0:
                angle = 2 * np.pi - angle
            return angle

        start_angle = angle_from_ref(v1)
        end_angle = angle_from_ref(v2)

        v1_unit = v1 / np.linalg.norm(v1)
        v2_unit = v2 / np.linalg.norm(v2)

        cross = np.cross(v1_unit, v2_unit)
        is_counter_clockwise = np.dot(cross, normal) > 0

        if not is_counter_clockwise:
            start_angle, end_angle = end_angle, start_angle
        start_angle = math.degrees(start_angle)
        end_angle = math.degrees(end_angle)

        return EllipseArc3D(
            start_point,
            end_point,
            center,
            mid_point,
            major_radius,
            minor_radius,
            normal,
            start_angle,
            end_angle,
            ref_vec,
            is_counter_clockwise,
        )

    def sample(self, num_points=5):
        num_points = max(num_points, 2)

        normal = self.normal / np.linalg.norm(self.normal)
        u = self.ref_vec / np.linalg.norm(self.ref_vec)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)

        start_angle = np.deg2rad(self.start_angle_ec)
        end_angle = np.deg2rad(self.end_angle_ec)
        if end_angle < start_angle:
            end_angle += 2 * np.pi

        angles = np.linspace(start_angle, end_angle, num_points)

        points = []
        for theta in angles:
            point = (
                self.major_radius * np.cos(theta) * u
                + self.minor_radius * np.sin(theta) * v
            )
            points.append(point + self.center)

        return np.array(points).tolist()

    def transform(self, movement=(0, 0, 0), scale=1):
        self.start_point = (self.start_point + movement) * scale
        self.end_point = (self.end_point + movement) * scale
        self.center = (self.center + movement) * scale
        self.mid_point = (self.mid_point + movement) * scale
        self.major_radius *= scale
        self.minor_radius *= scale

    def _json(self):
        arc_json = {
            "type": "3D Elliptical Arc",
            "start": [round(v, ROUND_JSON) for v in self.start_point],
            "end": [round(v, ROUND_JSON) for v in self.end_point],
            "via": [round(v, ROUND_JSON) for v in self.mid_point],
        }

        return arc_json


class ComplexCurve3D(Curve3D):
    def __init__(self, start_point, end_point, mid_point):
        super().__init__()
        self.start_point: np.ndarray = start_point
        self.end_point: np.ndarray = end_point
        self.mid_point: np.ndarray = mid_point

    @property
    def length(self):
        return -1

    @staticmethod
    def from_dict(stat):
        assert stat["type"] == "ComplexCurve3D"
        start_point = np.array(
            [
                stat["start_point"]["x"],
                stat["start_point"]["y"],
                stat["start_point"]["z"],
            ]
        )
        end_point = np.array(
            [stat["end_point"]["x"], stat["end_point"]["y"], stat["end_point"]["z"]]
        )
        mid_point = np.array(
            [stat["mid_point"]["x"], stat["mid_point"]["y"], stat["mid_point"]["z"]]
        )
        return ComplexCurve3D(start_point, end_point, mid_point)

    @staticmethod
    def from_occ(edge: Edge):
        occ_edge = edge.topods_shape()
        curve_handle, first_param, last_param = BRep_Tool.Curve(occ_edge)

        start_gp = gp_Pnt()
        end_gp = gp_Pnt()
        mid_gp = gp_Pnt()

        curve_handle.D0(first_param, start_gp)
        curve_handle.D0(last_param, end_gp)
        curve_handle.D0((first_param + last_param) / 2.0, mid_gp)

        def gp_pnt_to_np(gp_pnt):
            return np.array([gp_pnt.X(), gp_pnt.Y(), gp_pnt.Z()])

        start_point = gp_pnt_to_np(start_gp)
        end_point = gp_pnt_to_np(end_gp)
        mid_point = gp_pnt_to_np(mid_gp)

        return ComplexCurve3D(start_point, end_point, mid_point)

    def sample(self, num_points=5):
        return [
            self.start_point.tolist(),
            self.end_point.tolist(),
            self.mid_point.tolist(),
        ]

    def transform(self, movement=(0, 0, 0), scale=1):
        self.start_point = (self.start_point + movement) * scale
        self.end_point = (self.end_point + movement) * scale
        self.mid_point = (self.mid_point + movement) * scale

    def _json(self):
        line_json = {
            "type": "Complex 3D Curve",
            "start": [round(v, ROUND_JSON) for v in self.start_point],
            "end": [round(v, ROUND_JSON) for v in self.end_point],
            "via": [round(v, ROUND_JSON) for v in self.mid_point],
        }

        return line_json

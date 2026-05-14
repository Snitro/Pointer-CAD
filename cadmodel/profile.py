import cv2
import math
import numpy as np

from misc import ROUND_JSON, TOKEN, token_quantize, token_dequantize


class Curve:
    @staticmethod
    def from_dict(stat):
        pass

    def sample(self):
        pass

    def transform(self, movement=[0, 0], scale=1):
        pass

    def direction(self, from_start=True):
        pass

    def set_end_point(self, end_point):
        pass

    def reverse(self):
        pass

    def to_vector(self, match_curve_func=None, quant_bits=0):
        pass

    def _json(self):
        pass

    @property
    def curve_type(self):
        if isinstance(self, Line):
            return "line"
        elif isinstance(self, Circle):
            return "circle"
        elif isinstance(self, Arc):
            return "arc"
        else:
            raise NotImplementedError(
                "curve type not supported yet: {}".format(type(self))
            )

    @staticmethod
    def start_token():
        return [TOKEN.index("<|curve_start|>"), None]

    @staticmethod
    def from_vector(vector, project_point_func, quant_bits=0):
        items = [[]]
        for idx, item in enumerate(vector):
            if item[0] in [
                TOKEN.index("<|curve_start|>"),
                TOKEN.index("<|pointer_enable|>"),
                TOKEN.index("<|pointer_disable|>"),
            ]:
                items.append([(idx, -item[0])])
                items.append([])
            else:
                items[-1].append((idx, item[0]))

        curve_type = Curve
        extra_idx = None
        for extra_idx in reversed(range(len(items))):
            if len(items[extra_idx]) == 0:
                curve_type = Line
                break
            elif len(items[extra_idx]) == 1 and items[extra_idx][0][1] >= len(TOKEN):
                curve_type = Circle
                break
            elif (
                len(items[extra_idx]) == 2
                and items[extra_idx][0][1] >= len(TOKEN)
                and items[extra_idx][1][1]
                in [TOKEN.index("<|counter_clockwise|>"), TOKEN.index("<|clockwise|>")]
            ):
                curve_type = Arc
                break
        if curve_type == Curve:
            raise ValueError("Not enough elements to decode Curve from vector.")

        pointer = None
        pointer_idx = extra_idx - 1
        for pointer_idx in reversed(range(extra_idx)):
            if len(items[pointer_idx]) == 1 and items[pointer_idx][0][1] in [
                -TOKEN.index("<|pointer_enable|>"),
                -TOKEN.index("<|pointer_disable|>"),
            ]:
                pointer = items[pointer_idx][0][0]
                break
        if pointer is None:
            raise ValueError("Not enough elements to decode Curve from vector.")
        pointer = (
            vector[pointer][1][1]
            if vector[pointer][0] == TOKEN.index("<|pointer_enable|>")
            else None
        )

        start_point = None
        for start_point_idx in reversed(range(pointer_idx + 1)):
            if (
                len(items[start_point_idx]) == 2
                and items[start_point_idx][0][1] >= len(TOKEN)
                and items[start_point_idx][1][1] >= len(TOKEN)
            ):
                start_point = [
                    items[start_point_idx][0][1],
                    items[start_point_idx][1][1],
                ]
                break
        if start_point is None:
            raise ValueError("Not enough elements to decode Curve from vector.")
        start_point = token_dequantize(start_point, 0, 1, quant_bits)

        if pointer is not None:
            start_point = project_point_func(start_point, pointer)

        if curve_type == Line:
            return Line(np.array(start_point), None)
        elif curve_type == Circle:
            r = items[extra_idx][0][1]
            r = token_dequantize(r, 0, 1, quant_bits)
            return Circle(np.array(start_point), r)
        elif curve_type == Arc:
            a = items[extra_idx][0][1]
            a = token_dequantize(a, 0, 360, quant_bits)
            c = items[extra_idx][1][1] == TOKEN.index("<|counter_clockwise|>")
            return Arc(np.array(start_point), None, None, None, a, c)


class Line(Curve):
    def __init__(self, start_point, end_point):
        super(Line, self).__init__()
        self.start_point = start_point
        self.end_point = end_point

    @staticmethod
    def from_dict(stat):
        assert stat["type"] == "Line3D"
        start_point = np.array([stat["start_point"]["x"], stat["start_point"]["y"]])
        end_point = np.array([stat["end_point"]["x"], stat["end_point"]["y"]])
        return Line(start_point, end_point)

    def set_end_point(self, end_point):
        if np.allclose(self.start_point, end_point):
            raise ValueError(
                "Cannot construct Line: start point and end point are identical."
            )
        self.end_point = end_point

    def sample(self):
        return np.vstack(
            [
                np.array(self.start_point, dtype=np.float64),
                np.array(self.end_point, dtype=np.float64),
            ]
        )

    def transform(self, movement=[0, 0], scale=1):
        self.start_point = (self.start_point + movement) * scale
        self.end_point = (self.end_point + movement) * scale

    def direction(self, from_start=True):
        return self.end_point - self.start_point

    def reverse(self):
        self.start_point, self.end_point = self.end_point, self.start_point

    def to_vector(self, match_curve_func=None, quant_bits=0):
        start_point = token_quantize(self.start_point, 0, 1, quant_bits)
        start_point_pointer = (
            match_curve_func(self.start_point) if match_curve_func is not None else []
        )

        vec = [
            self.start_token(),
            [int(start_point[0]), None],
            [int(start_point[1]), None],
            [
                TOKEN.index(
                    "<|pointer_enable|>"
                    if len(start_point_pointer)
                    else "<|pointer_disable|>"
                ),
                start_point_pointer if len(start_point_pointer) else None,
            ],
        ]

        return vec

    def _json(self):
        return {
            "type": "Line",
            "start": [round(v, ROUND_JSON) for v in self.start_point],
            "end": [round(v, ROUND_JSON) for v in self.end_point],
        }


class Circle(Curve):
    def __init__(self, center, radius):
        super(Circle, self).__init__()
        self.center = center
        self.radius = radius

    @staticmethod
    def from_dict(stat):
        assert stat["type"] == "Circle3D"
        center = np.array([stat["center_point"]["x"], stat["center_point"]["y"]])
        radius = stat["radius"]
        return Circle(center, radius)

    def sample(self, step_angle=1):
        angles = np.deg2rad(np.arange(0, 360, step_angle))

        x = self.center[0] + self.radius * np.cos(angles)
        y = self.center[1] + self.radius * np.sin(angles)

        return np.column_stack((x, y)).astype(np.float64)

    def transform(self, movement=[0, 0], scale=1):
        self.center = (self.center + movement) * scale
        self.radius *= scale

    def to_vector(self, match_curve_func=None, quant_bits=0):
        center = token_quantize(self.center, 0, 1, quant_bits)
        center_pointer = (
            match_curve_func(self.center) if match_curve_func is not None else []
        )
        radius = token_quantize(self.radius, 0, 1, quant_bits)

        vec = [
            self.start_token(),
            [int(center[0]), None],
            [int(center[1]), None],
            [
                TOKEN.index(
                    "<|pointer_enable|>"
                    if len(center_pointer)
                    else "<|pointer_disable|>"
                ),
                center_pointer if len(center_pointer) else None,
            ],
            [radius, None],
        ]

        return vec

    def _json(self):
        circle_json = {
            "type": "Circle",
            "center": [round(v, ROUND_JSON) for v in self.center],
            "radius": round(self.radius, ROUND_JSON),
        }
        return circle_json


class Arc(Curve):
    def __init__(
        self, start_point, end_point, center, radius, sweep_angle, is_counter_clockwise
    ):
        super(Arc, self).__init__()
        self.start_point = start_point
        self.end_point = end_point
        self.center = center
        self.radius = radius
        self.sweep_angle = sweep_angle
        self.is_counter_clockwise = is_counter_clockwise

        if self.sweep_angle is not None:
            assert 0 < self.sweep_angle < 360, "sweep_angle must be in (0, 360) degrees"

    @property
    def mid_point(self):
        ref_vec = (
            self.start_point if self.is_counter_clockwise else self.end_point
        ) - self.center
        theta = np.deg2rad(self.sweep_angle / 2)

        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

        rotated_vec = rotation_matrix @ ref_vec

        mid_point = self.center + rotated_vec
        return mid_point

    @staticmethod
    def _angle_between(v1, v2):
        dot = np.dot(v1, v2)
        dot = np.clip(dot, -1.0, 1.0)
        return np.arccos(dot)

    @staticmethod
    def from_dict(stat):
        assert stat["type"] == "Arc3D"
        start_point = np.array([stat["start_point"]["x"], stat["start_point"]["y"]])
        end_point = np.array([stat["end_point"]["x"], stat["end_point"]["y"]])
        center = np.array([stat["center_point"]["x"], stat["center_point"]["y"]])
        radius = stat["radius"]
        start_angle = stat["start_angle"]
        sweep_angle = math.degrees(stat["end_angle"])
        ref_vec = np.array(
            [stat["reference_vector"]["x"], stat["reference_vector"]["y"]]
        )

        assert start_angle == 0, "start_angle must be 0"
        assert sweep_angle > 0, "end_angle must be positive"

        vec_start = start_point - center
        vec_end = end_point - center
        vec_start_unit = vec_start / np.linalg.norm(vec_start)
        vec_end_unit = vec_end / np.linalg.norm(vec_end)
        ref_vec_unit = ref_vec / np.linalg.norm(ref_vec)

        angle_with_start = Arc._angle_between(ref_vec_unit, vec_start_unit)
        angle_with_end = Arc._angle_between(ref_vec_unit, vec_end_unit)

        tolerance = math.radians(5)

        if angle_with_start < angle_with_end:
            if angle_with_start > tolerance:
                raise ValueError("Angle between ref_vec and start_vec is too large and inconsistent")
            is_counter_clockwise = True
        else:
            if angle_with_end > tolerance:
                raise ValueError("Angle between ref_vec and end_vec is too large and inconsistent")
            is_counter_clockwise = False

        return Arc(
            start_point, end_point, center, radius, sweep_angle, is_counter_clockwise
        )

    def set_end_point(self, end_point):
        if np.allclose(self.start_point, end_point):
            raise ValueError(
                "Cannot construct Line: start point and end point are identical."
            )

        self.end_point = end_point

        s2e_vector = self.end_point - self.start_point
        s2e_distance = np.linalg.norm(s2e_vector)
        self.radius = s2e_distance / 2 / math.sin(math.radians(self.sweep_angle / 2))

        if s2e_distance > 2 * self.radius:
            raise ValueError(
                "The distance between the two points is greater than the diameter; cannot form a circle with radius R."
            )

        mid = (self.start_point + self.end_point) / 2
        h = math.sqrt(self.radius**2 - (s2e_distance / 2) ** 2)

        normal = np.array([-s2e_vector[1], s2e_vector[0]]) / s2e_distance

        offset = h * normal
        if self.sweep_angle <= 180:
            self.center = (
                (mid + offset) if self.is_counter_clockwise else (mid - offset)
            )
        else:
            self.center = (
                (mid - offset) if self.is_counter_clockwise else (mid + offset)
            )

    def sample(self, step_angle=1):
        """
        Sample points on the arc, including start and end points.

        Args:
        step_angle - Sampling angular interval in degrees.

        Returns:
        numpy array (N, 2) containing start point, sampled points, and end point.
        """
        cx, cy = self.center
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        r = self.radius

        theta1 = np.arctan2(y1 - cy, x1 - cx)
        theta1_deg = np.rad2deg(theta1)

        direction = 1 if self.is_counter_clockwise else -1
        angles_deg = np.arange(step_angle, self.sweep_angle, step_angle) * direction
        angles_rad = np.deg2rad(angles_deg + theta1_deg)

        x = cx + r * np.cos(angles_rad)
        y = cy + r * np.sin(angles_rad)
        sampled_points = np.column_stack((x, y)).astype(np.float64)

        result = np.vstack(
            [
                np.array(self.start_point, dtype=np.float64),
                sampled_points,
                np.array(self.end_point, dtype=np.float64),
            ]
        )

        return result

    def transform(self, movement=[0, 0], scale=1):
        self.start_point = (self.start_point + movement) * scale
        self.end_point = (self.end_point + movement) * scale
        self.center = (self.center + movement) * scale
        self.radius *= scale

    def direction(self, from_start=True):
        if from_start:
            return self.mid_point - self.start_point
        else:
            return self.end_point - self.mid_point

    def reverse(self):
        self.start_point, self.end_point, self.is_counter_clockwise = (
            self.end_point,
            self.start_point,
            not self.is_counter_clockwise,
        )

    def to_vector(self, match_curve_func=None, quant_bits=0):
        start_point = token_quantize(self.start_point, 0, 1, quant_bits)
        start_point_pointer = (
            match_curve_func(self.start_point) if match_curve_func is not None else []
        )
        sweep_angle = token_quantize(self.sweep_angle % 360, 0, 360, quant_bits)

        vec = [
            self.start_token(),
            [int(start_point[0]), None],
            [int(start_point[1]), None],
            [
                TOKEN.index(
                    "<|pointer_enable|>"
                    if len(start_point_pointer)
                    else "<|pointer_disable|>"
                ),
                start_point_pointer if len(start_point_pointer) else None,
            ],
            [sweep_angle, None],
            [
                TOKEN.index(
                    "<|counter_clockwise|>"
                    if self.is_counter_clockwise
                    else "<|clockwise|>"
                ),
                None,
            ],
        ]

        return vec

    def _json(self):
        arc_json = {
            "type": "Arc",
            "start": [round(v, ROUND_JSON) for v in self.start_point],
            "end": [round(v, ROUND_JSON) for v in self.end_point],
            "center": [round(v, ROUND_JSON) for v in self.center],
            "radius": round(self.radius, ROUND_JSON),
            "sweep_angle": round(self.sweep_angle, ROUND_JSON),
            "is_counter_clockwise": self.is_counter_clockwise,
        }

        return arc_json


class Loop(object):
    """Sketch loop, a sequence of connected curves."""

    def __init__(self, curves: list[Curve]):
        self.curves = curves

        self.reorder()

    @property
    def bbox(self):
        all_points = np.vstack([curve.sample() for curve in self.curves])
        min_point = np.min(all_points, axis=0)
        max_point = np.max(all_points, axis=0)
        return np.array([max_point, min_point], dtype=np.float64)

    @staticmethod
    def start_token():
        return [TOKEN.index("<|loop_start|>"), None]

    @staticmethod
    def construct_curve_from_dict(stat):
        if stat["type"] == "Line3D":
            return Line.from_dict(stat)
        elif stat["type"] == "Circle3D":
            return Circle.from_dict(stat)
        elif stat["type"] == "Arc3D":
            return Arc.from_dict(stat)
        else:
            raise NotImplementedError(
                "curve type not supported yet: {}".format(stat["type"])
            )

    @staticmethod
    def from_dict(stat):
        all_curves = [
            Loop.construct_curve_from_dict(item) for item in stat["profile_curves"]
        ]
        this_loop = Loop(all_curves)
        return this_loop

    def reorder(self):
        """Reorder curves to start from the left-most point and run counterclockwise."""
        if len(self.curves) <= 1:
            return

        if np.allclose(
            self.curves[0].start_point, self.curves[1].start_point
        ) or np.allclose(self.curves[0].start_point, self.curves[1].end_point):
            self.curves[0].reverse()

        start_curve_idx = 0
        sx, sy = self.curves[0].start_point

        for i, curve in enumerate(self.curves):
            if i < len(self.curves) - 1 and np.allclose(
                curve.end_point, self.curves[i + 1].end_point
            ):
                self.curves[i + 1].reverse()
            if round(curve.start_point[0], 6) < round(sx, 6) or (
                round(curve.start_point[0], 6) == round(sx, 6)
                and round(curve.start_point[1], 6) < round(sy, 6)
            ):
                start_curve_idx = i
                sx, sy = curve.start_point

        self.curves = self.curves[start_curve_idx:] + self.curves[:start_curve_idx]

        if isinstance(self.curves[0], Circle) or isinstance(
            self.curves[-1], Circle
        ):  # FIXME: hard-coded
            return

        start_vec = self.curves[0].direction()
        end_vec = self.curves[-1].direction(from_start=False)
        if np.cross(end_vec, start_vec) <= 0:
            for curve in self.curves:
                curve.reverse()
            self.curves.reverse()

    def to_vector(self, match_curve_func=None, quant_bits=0):
        vec = [self.start_token()]
        vec.extend(
            [
                v
                for curve in self.curves
                for v in curve.to_vector(match_curve_func, quant_bits)
            ]
        )
        return vec

    @staticmethod
    def from_vector(vector, project_point_func, quant_bits=0, strict=True):
        CV_positions = [i for i, x in enumerate(vector) if x == Curve.start_token()]

        curves: list[Curve] = []
        curve_start = 0
        while curve_start is not None:
            curve_end = curve_start

            while curve_end is not None:
                try:
                    curve_end = (
                        curve_end + 1 if curve_end + 1 < len(CV_positions) else None
                    )
                    curve_start_idx = CV_positions[curve_start]
                    curve_end_idx = (
                        CV_positions[curve_end] if curve_end is not None else None
                    )
                    curve_vector = vector[curve_start_idx + 1 : curve_end_idx]
                    curve = Curve.from_vector(
                        curve_vector, project_point_func, quant_bits
                    )

                    if not isinstance(curve, Circle):
                        for prev_curve in reversed(curves):
                            if not isinstance(prev_curve, Circle):
                                prev_curve.set_end_point(curve.start_point)
                                break

                    curves.append(curve)
                    break
                except Exception as e:
                    if strict:
                        raise e
            else:
                curve_start = (
                    curve_start + 1 if curve_start + 1 < len(CV_positions) else None
                )
                continue

            curve_start = curve_end

        start_point = None
        for start_curve in curves:
            if not isinstance(start_curve, Circle):
                start_point = start_curve.start_point
                break
        for rev_idx in range(len(curves) - 1, -1, -1):
            end_curve = curves[rev_idx]
            if not isinstance(end_curve, Circle):
                try:
                    end_curve.set_end_point(start_point)
                    break
                except Exception as e:
                    if strict:
                        raise e
                    else:
                        curves.pop(rev_idx)

        points = []
        for curve in curves:
            if isinstance(curve, Circle):
                return Loop(curves)
            points.append(curve.sample())

        points = np.vstack(points).astype(np.float32)
        if np.allclose(points[:, 0], points[0, 0]):
            raise ValueError("All x values are identical; x does not change.")
        if np.allclose(points[:, 1], points[0, 1]):
            raise ValueError("All y values are identical; y does not change.")

        return Loop(curves)

    def _json(self):
        loop_json = {
            f"curve_{idx + 1}": curve._json() for idx, curve in enumerate(self.curves)
        }
        return loop_json


class Profile(object):

    def __init__(self, loops: list[Loop]):
        self.loops = loops

        loop_bboxes = [loop.bbox for loop in loops]

        all_points = np.vstack(loop_bboxes)
        self.bbox = np.array(
            [np.max(all_points, axis=0), np.min(all_points, axis=0)], dtype=np.float64
        )

        loop_outer_counts = []
        for bbox in loop_bboxes:
            count = 0
            if np.allclose(bbox[0][0], self.bbox[0][0]):
                count += 1  # Right side
            if np.allclose(bbox[0][1], self.bbox[0][1]):
                count += 1  # Top side
            if np.allclose(bbox[1][0], self.bbox[1][0]):
                count += 1  # Left side
            if np.allclose(bbox[1][1], self.bbox[1][1]):
                count += 1  # Bottom side

            loop_outer_counts.append(count)

        self.loops = [
            loop
            for _, loop in sorted(
                zip(loop_outer_counts, self.loops), key=lambda x: x[0], reverse=True
            )
        ]

    @staticmethod
    def start_token():
        return [TOKEN.index("<|profile_start|>"), None]

    @staticmethod
    def from_dict(stat):
        all_loops = [Loop.from_dict(item) for item in stat["loops"]]
        return Profile(all_loops)

    def to_vector(self, match_curve_func=None, quant_bits=0):
        vec = [self.start_token()]
        vec.extend(
            [
                v
                for loop in self.loops
                for v in loop.to_vector(match_curve_func, quant_bits)
            ]
        )
        return vec

    @staticmethod
    def from_vector(vector, project_point_func, quant_bits=0, strict=True):
        loops: list[Loop] = []
        LP_positions = list(
            reversed([i for i, x in enumerate(vector) if x == Loop.start_token()])
        )
        loop_end = None
        for loop_start in LP_positions:
            try:
                loop_vector = vector[loop_start + 1 : loop_end]
                loop = Loop.from_vector(
                    loop_vector, project_point_func, quant_bits, strict
                )
                if len(loop.curves) == 0:
                    if strict:
                        raise ValueError(
                            "Not enough elements to decode Loop from vector."
                        )
                    else:
                        continue

                loops.append(loop)
                loop_end = loop_start
            except Exception as e:
                if strict:
                    raise e

        if len(loops) == 0:
            raise ValueError("Not enough elements to decode Profile from vector.")

        return Profile(loops)

    def _json(self):
        face_json = {}
        for i, loop in enumerate(self.loops):
            face_json[f"loop_{i+1}"] = loop._json()

        return face_json

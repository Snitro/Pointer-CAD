import cv2
import math
import numpy as np
from occwl.solid import Solid
from scipy.spatial.transform import Rotation as R


class CoordinateSystem(object):

    def __init__(self, rotation: R, origin, scale=1) -> None:
        self.rotation = rotation
        self.origin = origin
        self.scale = scale

    @property
    def normal(self):
        z_axis = self.rotation.as_matrix()[2]
        normal = z_axis / np.linalg.norm(z_axis)
        return normal

    @staticmethod
    def from_dict(stat):
        origin = np.array(
            [stat["origin"]["x"], stat["origin"]["y"], stat["origin"]["z"]]
        )
        x_axis_3d = np.array(
            [stat["x_axis"]["x"], stat["x_axis"]["y"], stat["x_axis"]["z"]]
        )
        y_axis_3d = np.array(
            [stat["y_axis"]["x"], stat["y_axis"]["y"], stat["y_axis"]["z"]]
        )
        z_axis_3d = np.array(
            [stat["z_axis"]["x"], stat["z_axis"]["y"], stat["z_axis"]["z"]]
        )
        rotation = R.from_matrix(np.vstack((x_axis_3d, y_axis_3d, z_axis_3d)))

        coord = CoordinateSystem(rotation, origin)

        return coord

    def world2relative(self, position):
        return self.rotation.apply(position - self.origin) / self.scale

    def relative2world(self, position):
        position = np.asarray(position)

        if position.shape[-1] == 2:
            zeros = np.zeros_like(position[..., :1])
            position = np.concatenate([position, zeros], axis=-1)
        return self.rotation.inv().apply(position * self.scale) + self.origin

    def transform(self, movement=(0, 0, 0), scale=1):
        self.origin = (self.origin + movement) * scale

    @staticmethod
    def _calculate_axis_intersections(point=None, normal=None):
        """
        Compute plane intersections with the x, y, and z axes.

        Args:
        - point: (x0, y0, z0), a point on the plane.
        - normal: (a, b, c), plane normal vector.

        Returns:
        - (x_axis_intersect, y_axis_intersect, z_axis_intersect)
        Each value is the non-zero coordinate of the axis intersection,
        or None if no intersection exists.
        """
        x0, y0, z0 = point
        a, b, c = normal

        d = -(a * x0 + b * y0 + c * z0)

        x_axis_intersect = None
        if a != 0:
            x_axis_intersect = -d / a

        y_axis_intersect = None
        if b != 0:
            y_axis_intersect = -d / b

        z_axis_intersect = None
        if c != 0:
            z_axis_intersect = -d / c

        return x_axis_intersect, y_axis_intersect, z_axis_intersect

    def _axis_intersections(self):
        return CoordinateSystem._calculate_axis_intersections(self.origin, self.normal)

    def _calculate_rotation_angle(self, base, target):
        """
        Compute counterclockwise rotation angle from base to target.

        Args:
            base, target: numpy arrays representing 3D vectors.

        Returns:
            Counterclockwise angle in degrees (float).
        """
        base_unit = base / np.linalg.norm(base)
        target_unit = target / np.linalg.norm(target)

        dot = np.clip(np.dot(base_unit, target_unit), -1.0, 1.0)
        angle = np.arccos(dot)

        cross = np.cross(base_unit, target_unit)
        direction = np.dot(cross, self.normal)

        if direction < 0:
            angle = 2 * np.pi - angle

        return np.degrees(angle)

    def to_vector(self, quant_bits=0):
        from misc import TOKEN, token_quantize

        x0, y0, z0 = self._axis_intersections()
        x, y, z = self.origin
        nx, ny, nz = self.normal

        abs_components = [("x", abs(nx), nx), ("y", abs(ny), ny), ("z", abs(nz), nz)]

        sorted_components = sorted(
            abs_components, key=lambda item: item[1], reverse=True
        )

        results = []

        for axis_name, abs_normal, normal in sorted_components:
            if normal == 0:
                continue

            if axis_name == "x":
                base_vec = np.array((0, 1, 0)) if normal > 0 else np.array((0, 0, 1))
                if normal > 0 and y0 is not None:
                    base_vec = np.array((0, y0, 0)) - np.array((x0, 0, 0))
                    base_vec *= -1 if y0 < 0 else 1
                elif normal < 0 and z0 is not None:
                    base_vec = np.array((0, 0, z0)) - np.array((x0, 0, 0))
                    base_vec *= -1 if z0 < 0 else 1

                angle = self._calculate_rotation_angle(
                    base_vec, self.relative2world((1, 0, 0)) - self.origin
                )
                if normal > 0:
                    results.append((TOKEN.index("<|direction_x+|>"), y, z, angle))
                else:
                    results.append((TOKEN.index("<|direction_x-|>"), z, y, angle))
            elif axis_name == "y":
                base_vec = np.array((0, 0, 1)) if normal > 0 else np.array((1, 0, 0))
                if normal > 0 and z0 is not None:
                    base_vec = np.array((0, 0, z0)) - np.array((0, y0, 0))
                    base_vec *= -1 if z0 < 0 else 1
                elif normal < 0 and x0 is not None:
                    base_vec = np.array((x0, 0, 0)) - np.array((0, y0, 0))
                    base_vec *= -1 if x0 < 0 else 1

                angle = self._calculate_rotation_angle(
                    base_vec, self.relative2world((1, 0, 0)) - self.origin
                )
                if normal > 0:
                    results.append((TOKEN.index("<|direction_y+|>"), z, x, angle))
                else:
                    results.append((TOKEN.index("<|direction_y-|>"), x, z, angle))
            elif axis_name == "z":
                base_vec = np.array((1, 0, 0)) if normal > 0 else np.array((0, 1, 0))
                if normal > 0 and x0 is not None:
                    base_vec = np.array((x0, 0, 0)) - np.array((0, 0, z0))
                    base_vec *= -1 if x0 < 0 else 1
                elif normal < 0 and y0 is not None:
                    base_vec = np.array((0, y0, 0)) - np.array((0, 0, z0))
                    base_vec *= -1 if y0 < 0 else 1

                angle = self._calculate_rotation_angle(
                    base_vec, self.relative2world((1, 0, 0)) - self.origin
                )
                if normal > 0:
                    results.append((TOKEN.index("<|direction_z+|>"), x, y, angle))
                else:
                    results.append((TOKEN.index("<|direction_z-|>"), y, x, angle))

        vectors = []
        for result in results:
            u = token_quantize(result[1], -1, 1, quant_bits)
            v = token_quantize(result[2], -1, 1, quant_bits)
            theta = token_quantize((result[3] + 360) % 360, 0, 360, quant_bits)
            s = token_quantize(self.scale, 0, 2, quant_bits)
            vec = [[result[0], None], [u, None], [v, None], [theta, None], [s, None]]

            vectors.append(vec)

        return vectors

    @staticmethod
    def _calculate_plane_origin(x, y, z, normal: np.array, point: np.array):
        """
        x, y, z: float or None, use None for the unknown coordinate.
        normal: np.array, normal vector [A, B, C].
        point: np.array, point on the plane [x0, y0, z0].

        Returns:
            Solved (x, y, z).
        """
        A, B, C = normal
        x0, y0, z0 = point

        known_coords = [c is not None for c in (x, y, z)]
        if known_coords.count(True) != 2:
            raise ValueError("Exactly two coordinates must be provided, and one must be None")

        if x is None:
            if A == 0:
                raise ValueError("Cannot solve x because A = 0")
            x = float(x0 - (B * (y - y0) + C * (z - z0)) / A)

        elif y is None:
            if B == 0:
                raise ValueError("Cannot solve y because B = 0")
            y = float(y0 - (A * (x - x0) + C * (z - z0)) / B)

        elif z is None:
            if C == 0:
                raise ValueError("Cannot solve z because C = 0")
            z = float(z0 - (A * (x - x0) + B * (y - y0)) / C)

        return x, y, z

    @staticmethod
    def _rotate_vector_by_angle(
        base: np.ndarray, normal: np.ndarray, angle_deg: float
    ) -> np.ndarray:
        """
        Rotate base around normal by the given counterclockwise angle (degrees).
        :param base: 3D vector, np.array([x, y, z])
        :param normal: Rotation axis (normal vector), does not need normalization
        :param angle_deg: Rotation angle in degrees
        :return: Rotated vector np.ndarray
        """
        angle_rad = np.deg2rad(angle_deg)
        k = normal / np.linalg.norm(normal)

        base_rot = (
            base * np.cos(angle_rad)
            + np.cross(k, base) * np.sin(angle_rad)
            + k * np.dot(k, base) * (1 - np.cos(angle_rad))
        )

        return base_rot

    @staticmethod
    def from_vector(vector, ref_normal, ref_origin, quant_bits=0):
        from misc import TOKEN, token_dequantize

        dir_normal = {
            "<|direction_x+|>": np.array([1, 0, 0]),
            "<|direction_x-|>": np.array([-1, 0, 0]),
            "<|direction_y+|>": np.array([0, 1, 0]),
            "<|direction_y-|>": np.array([0, -1, 0]),
            "<|direction_z+|>": np.array([0, 0, 1]),
            "<|direction_z-|>": np.array([0, 0, -1]),
        }
        dir_idx = [TOKEN.index(token_name) for token_name in dir_normal]

        DIR_positions = [i for i, x in enumerate(vector) if x[0] in dir_idx]
        if len(DIR_positions) == 0:
            raise ValueError(
                "Not enough elements to decode Coordinate System from vector."
            )

        for d_idx in DIR_positions:
            value_list = []
            for value_idx in range(d_idx + 1, len(vector)):
                if vector[value_idx][0] >= len(TOKEN):
                    value_list.append(vector[value_idx][0])
            u = token_dequantize(value_list[0], -1, 1, quant_bits)
            v = token_dequantize(value_list[1], -1, 1, quant_bits)
            theta = token_dequantize(value_list[2], 0, 360, quant_bits)
            s = token_dequantize(value_list[3], 0, 2, quant_bits)

            direction = TOKEN[vector[d_idx][0]]
            dir_dot = np.dot(ref_normal, dir_normal[direction])

            if dir_dot == 0 or s == 0:
                continue

            normal = ref_normal if dir_dot > 0 else -ref_normal
            x0, y0, z0 = CoordinateSystem._calculate_axis_intersections(
                ref_origin, normal
            )
            origin = [None, None, None]
            rotation_ref = None
            if direction == "<|direction_x+|>":
                origin = [None, u, v]
                if y0 is not None:
                    rotation_ref = np.array((0, y0, 0)) - np.array((x0, 0, 0))
                    rotation_ref *= -1 if y0 < 0 else 1
                else:
                    rotation_ref = np.array((0, 1, 0))
            elif direction == "<|direction_x-|>":
                origin = [None, v, u]
                if z0 is not None:
                    rotation_ref = np.array((0, 0, z0)) - np.array((x0, 0, 0))
                    rotation_ref *= -1 if z0 < 0 else 1
                else:
                    rotation_ref = np.array((0, 0, 1))
            elif direction == "<|direction_y+|>":
                origin = [v, None, u]
                if z0 is not None:
                    rotation_ref = np.array((0, 0, z0)) - np.array((0, y0, 0))
                    rotation_ref *= -1 if z0 < 0 else 1
                else:
                    rotation_ref = np.array((0, 0, 1))
            elif direction == "<|direction_y-|>":
                origin = [u, None, v]
                if x0 is not None:
                    rotation_ref = np.array((x0, 0, 0)) - np.array((0, y0, 0))
                    rotation_ref *= -1 if x0 < 0 else 1
                else:
                    rotation_ref = np.array((1, 0, 0))
            elif direction == "<|direction_z+|>":
                origin = [u, v, None]
                if x0 is not None:
                    rotation_ref = np.array((x0, 0, 0)) - np.array((0, 0, z0))
                    rotation_ref *= -1 if x0 < 0 else 1
                else:
                    rotation_ref = np.array((1, 0, 0))
            elif direction == "<|direction_z-|>":
                origin = [v, u, None]
                if y0 is not None:
                    rotation_ref = np.array((0, y0, 0)) - np.array((0, 0, z0))
                    rotation_ref *= -1 if y0 < 0 else 1
                else:
                    rotation_ref = np.array((0, 1, 0))

            origin = CoordinateSystem._calculate_plane_origin(
                *origin, normal=normal, point=ref_origin
            )
            x_axis_3d = CoordinateSystem._rotate_vector_by_angle(
                rotation_ref, normal, theta
            )
            y_axis_3d = np.cross(normal, x_axis_3d)
            rotation = R.from_matrix(np.vstack((x_axis_3d, y_axis_3d, normal)))

            return CoordinateSystem(rotation, origin, s)

        raise ValueError("Not enough elements to decode Coordinate System from vector.")

    def _json(self):
        from misc import ROUND_JSON

        normal = self.normal

        plane = None
        rotation = [0, 0, 0]
        if np.allclose(normal, [0, 0, -1], atol=1e-6):
            plane = "Bottom"
        elif np.allclose(normal, [1, 0, 0], atol=1e-6):
            plane = "Right"
        elif np.allclose(normal, [-1, 0, 0], atol=1e-6):
            plane = "Left"
        elif np.allclose(normal, [0, -1, 0], atol=1e-6):
            plane = "Front"
        elif np.allclose(normal, [0, 1, 0], atol=1e-6):
            plane = "Back"
        else:
            plane = "Top"
            rotation = self.rotation.as_euler("zyx", degrees=True)

        json_data = {
            "reference_plane": plane,
            "rotation": [round(r, ROUND_JSON) for r in rotation],
            "position": [round(p, ROUND_JSON) for p in self.origin],
        }
        if self.scale != 1:
            json_data["scale"] = self.scale
        return json_data

import numpy as np

from .curve3d import Curve3D


class ModelFeature(object):
    def __init__(self, index=0, bbox=None, name=None):
        self.bbox: np.ndarray = bbox
        self.name: str = name
        self.index: int = index

    @staticmethod
    def parse_bounding_box(entity: dict):
        if "bounding_box" in entity:
            bbox_info = entity["bounding_box"]
            max_point = np.array(
                [
                    bbox_info["max_point"]["x"],
                    bbox_info["max_point"]["y"],
                    bbox_info["max_point"]["z"],
                ]
            )
            min_point = np.array(
                [
                    bbox_info["min_point"]["x"],
                    bbox_info["min_point"]["y"],
                    bbox_info["min_point"]["z"],
                ]
            )
            bbox = np.stack([max_point, min_point], axis=0)
            return bbox
        else:
            return None

    @staticmethod
    def get_3d_curve_by_edge(edge, body, feature, all_stat):
        curve_data = all_stat["features"][feature]["entities"][body]["curves"][edge]
        return Curve3D.from_dict(curve_data)

    @staticmethod
    def get_3d_curve_by_face(face, body, feature, all_stat):
        face_data = all_stat["features"][feature]["entities"][body]["faces"][face]
        for edge, edge_info in face_data["loops"].items():
            yield ModelFeature.get_3d_curve_by_edge(
                edge, edge_info["body"], edge_info["feature"], all_stat
            )

    def transform(self, movement=(0, 0, 0), scale=1):
        if self.bbox is not None:
            for i in range(2):
                self.bbox[i] += movement
            for i in range(2):
                self.bbox[i] *= scale

    def build_part(self, pre_model=None):
        pass

    def to_vector(self, built_solid=None, quant_bits=0):
        pass

    @staticmethod
    def from_vector(vector, quant_bits=0):
        pass

    def _json(self):
        pass

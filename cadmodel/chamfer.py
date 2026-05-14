import math
import numpy as np
from occwl.edge import Edge
from occwl.solid import Solid
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeChamfer
from OCC.Core.TopoDS import TopoDS_Compound

from .curve3d import Curve3D
from .modelfeature import ModelFeature
from misc import (
    ROUND_JSON,
    find_edge_by_points_and_length,
    TOKEN,
    token_quantize,
    token_dequantize,
)


class Chamfer(ModelFeature):
    def __init__(
        self,
        index,
        edges: list[Curve3D],
        distance,
        tangent_chain=True,
        bbox=None,
        name=None,
    ):
        super().__init__(index, bbox, name)

        self.edges = edges
        self.distance = distance
        self.tangent_chain = tangent_chain

    @staticmethod
    def start_token():
        return [TOKEN.index("<|chamfer_start|>"), None]

    @staticmethod
    def from_dict(all_stat, fillet_id, index):
        chamfer_entity = all_stat["features"][fillet_id]
        assert chamfer_entity["type"] == "ChamferFeature"
        assert chamfer_entity["chamfer_method"] == "OffsetMeasurement"
        assert chamfer_entity["tangent_chain"]

        edges = []
        for edge, edge_info in chamfer_entity["edges"].items():
            edges.append(
                ModelFeature.get_3d_curve_by_edge(
                    edge, edge_info["body"], edge_info["feature"], all_stat
                )
            )
        for face, face_info in chamfer_entity["faces"].items():
            edges.extend(
                list(
                    ModelFeature.get_3d_curve_by_face(
                        face, face_info["body"], face_info["feature"], all_stat
                    )
                )
            )

        distance = chamfer_entity["distance"]["value"]
        distance1 = chamfer_entity["distance_one"]["value"]
        distance2 = chamfer_entity["distance_two"]["value"]
        angle = chamfer_entity["angle"]["value"]
        tangent_chain = chamfer_entity["tangent_chain"]

        if chamfer_entity["chamfer_method"] == "TwoDistancesChamferType":
            assert distance1 == distance2
            distance = distance1
        elif chamfer_entity["chamfer_method"] == "DistanceAndAngleChamferType":
            assert np.allclose(math.degrees(angle), 45)

        bbox = ModelFeature.parse_bounding_box(chamfer_entity)

        name = chamfer_entity["name"]

        return Chamfer(index, edges, distance, tangent_chain, bbox, name)

    def transform(self, movement=(0, 0, 0), scale=1):
        super().transform(movement, scale)

        for edge in self.edges:
            edge.transform(movement, scale)
        self.distance *= scale

    def build_part(self, pre_model: TopoDS_Compound = None):
        if pre_model is None:
            raise ValueError("pre_model must be provided.")

        matched_edges = []
        for edge in self.edges:
            matched_edge = find_edge_by_points_and_length(
                pre_model, edge.sample(), edge.length
            )
            if matched_edge is not None:
                matched_edges.append(matched_edge)

        def try_chamfer_with_distance(scale_factor):
            chamfer_maker = BRepFilletAPI_MakeChamfer(pre_model)
            for edge in matched_edges:
                chamfer_maker.Add(self.distance * scale_factor, edge)
            chamfer_maker.Build()
            return chamfer_maker if chamfer_maker.IsDone() else None

        # Try original radius
        fillet_maker = try_chamfer_with_distance(1.0)
        # Try slightly smaller radius if failed
        if fillet_maker is None:
            fillet_maker = try_chamfer_with_distance(0.999999)
        # Try even smaller radius if still failed
        if fillet_maker is None:
            fillet_maker = try_chamfer_with_distance(0.9999)
        # Try even smaller radius if still failed
        if fillet_maker is None:
            fillet_maker = try_chamfer_with_distance(0.99)

        if fillet_maker is None:
            raise RuntimeError(
                "Fillet operation failed. Possibly due to invalid edges or topology."
            )

        return fillet_maker.Shape()

    def to_vector(self, built_solid: Solid = None, quant_bits=0, flatten=False):
        vec = [
            Chamfer.start_token(),
            [token_quantize(self.distance, 0, 2, quant_bits), None],
        ]

        matched_edges = []
        for edge in self.edges:
            matched_edge = find_edge_by_points_and_length(
                built_solid.topods_shape(), edge.sample(), edge.length
            )
            if matched_edge is not None and matched_edge not in matched_edges:
                matched_edges.append(matched_edge)

        if len(matched_edges) == 0:
            raise ValueError("No matched edges found in built_solid for Chamfer.")

        if flatten:
            for edge in matched_edges:
                vec.append([TOKEN.index("<|pointer_enable|>"), Edge(edge)])
        else:
            vec.append(
                [
                    TOKEN.index("<|pointer_enable|>"),
                    [Edge(edge) for edge in matched_edges],
                ]
            )

        return vec

    @staticmethod
    def from_vector(vector, quant_bits=0, strict=True):
        ST_positions = list(
            reversed([i for i, x in enumerate(vector) if x == Chamfer.start_token()])
        )
        for idx in ST_positions:
            chamfer_vector = vector[idx + 1 :]
            if len(chamfer_vector) < 2:
                if strict:
                    raise ValueError(
                        "Not enough elements to decode chamfer from vector."
                    )
                else:
                    continue

            D_positions = list(
                reversed(
                    [i for i, x in enumerate(chamfer_vector) if x[0] >= len(TOKEN)]
                )
            )
            for d_idx in D_positions:
                distance = token_dequantize(chamfer_vector[d_idx][0], 0, 2, quant_bits)
                edges = []
                for edge_vector in chamfer_vector[d_idx + 1 :]:
                    if edge_vector[0] == TOKEN.index("<|pointer_enable|>"):
                        edge_pointer: Edge = edge_vector[1][1]
                        edges.append(Curve3D.from_occ(edge_pointer))

                if len(edges) > 0:
                    return Chamfer(-1, edges, distance)

        raise ValueError("Not enough elements to decode chamfer from vector.")

    def _json(self):
        edges = {
            f"edge_{idx + 1}": curve._json() for idx, curve in enumerate(self.edges)
        }

        chamfer_json = {
            "chamfer_edges": edges,
            "chamfer_distance": round(self.distance, ROUND_JSON),
            "chamfer_tangent_chain": self.tangent_chain,
        }

        if self.name is not None:
            chamfer_json_name = {"name": self.name}
            chamfer_json_name.update(chamfer_json)
            chamfer_json = chamfer_json_name

        return chamfer_json

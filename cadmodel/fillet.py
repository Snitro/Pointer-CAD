from occwl.edge import Edge
from occwl.solid import Solid
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet
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


class Fillet(ModelFeature):
    def __init__(
        self,
        index,
        edges: list[Curve3D],
        radius,
        tangent_chain=True,
        bbox=None,
        name=None,
    ):
        super().__init__(index, bbox, name)

        self.edges = edges
        self.radius = radius
        self.tangent_chain = tangent_chain

    @staticmethod
    def start_token():
        return [TOKEN.index("<|fillet_start|>"), None]

    @staticmethod
    def from_dict(all_stat, fillet_id, index):
        fillet_entity = all_stat["features"][fillet_id]
        assert fillet_entity["type"] == "FilletFeature"
        assert fillet_entity["blend_control_type"] == "RADIUS"
        assert fillet_entity["tangent_chain"]

        edges = []
        for edge, edge_info in fillet_entity["edges"].items():
            edges.append(
                ModelFeature.get_3d_curve_by_edge(
                    edge, edge_info["body"], edge_info["feature"], all_stat
                )
            )
        for face, face_info in fillet_entity["faces"].items():
            edges.extend(
                list(
                    ModelFeature.get_3d_curve_by_face(
                        face, face_info["body"], face_info["feature"], all_stat
                    )
                )
            )

        radius = fillet_entity["radius"]["value"]
        tangent_chain = fillet_entity["tangent_chain"]

        bbox = ModelFeature.parse_bounding_box(fillet_entity)

        name = fillet_entity["name"]

        return Fillet(index, edges, radius, tangent_chain, bbox, name)

    def transform(self, movement=(0, 0, 0), scale=1):
        super().transform(movement, scale)

        for edge in self.edges:
            edge.transform(movement, scale)
        self.radius *= scale

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

        def try_fillet_with_radius(scale_factor):
            fillet_maker = BRepFilletAPI_MakeFillet(pre_model)
            for edge in matched_edges:
                fillet_maker.Add(self.radius * scale_factor, edge)
            fillet_maker.Build()
            return fillet_maker if fillet_maker.IsDone() else None

        # Try original radius
        fillet_maker = try_fillet_with_radius(1.0)
        # Try slightly smaller radius if failed
        if fillet_maker is None:
            fillet_maker = try_fillet_with_radius(0.999999)
        # Try even smaller radius if still failed
        if fillet_maker is None:
            fillet_maker = try_fillet_with_radius(0.9999)
        # Try even smaller radius if still failed
        if fillet_maker is None:
            fillet_maker = try_fillet_with_radius(0.99)

        if fillet_maker is None:
            raise RuntimeError(
                "Fillet operation failed. Possibly due to invalid edges or topology."
            )

        return fillet_maker.Shape()

    def to_vector(self, built_solid: Solid = None, quant_bits=0, flatten=False):
        vec = [
            Fillet.start_token(),
            [token_quantize(self.radius, 0, 2, quant_bits), None],
        ]

        matched_edges = []
        for edge in self.edges:
            matched_edge = find_edge_by_points_and_length(
                built_solid.topods_shape(), edge.sample(), edge.length
            )
            if matched_edge is not None and matched_edge not in matched_edges:
                matched_edges.append(matched_edge)

        if len(matched_edges) == 0:
            raise ValueError("No matched edges found in built_solid for Fillet.")

        if flatten:
            for edge in matched_edges:
                vec.append([TOKEN.index("<|pointer_enable|>"), [Edge(edge)]])
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
            reversed([i for i, x in enumerate(vector) if x == Fillet.start_token()])
        )
        for idx in ST_positions:
            fillet_vector = vector[idx + 1 :]
            if len(fillet_vector) < 2:
                if strict:
                    raise ValueError(
                        "Not enough elements to decode fillet from vector."
                    )
                else:
                    continue

            D_positions = list(
                reversed([i for i, x in enumerate(fillet_vector) if x[0] >= len(TOKEN)])
            )
            for d_idx in D_positions:
                distance = token_dequantize(fillet_vector[d_idx][0], 0, 2, quant_bits)
                edges = []
                for edge_vector in fillet_vector[d_idx + 1 :]:
                    if edge_vector[0] == TOKEN.index("<|pointer_enable|>"):
                        edge_pointer: Edge = edge_vector[1][1]
                        edges.append(Curve3D.from_occ(edge_pointer))

                if len(edges) > 0:
                    return Fillet(-1, edges, distance)

        raise ValueError("Not enough elements to decode fillet from vector.")

    def _json(self):
        edges = {
            f"edge_{idx + 1}": curve._json() for idx, curve in enumerate(self.edges)
        }

        fillet_json = {
            "fillet_edges": edges,
            "fillet_radius": round(self.radius, ROUND_JSON),
            "fillet_tangent_chain": self.tangent_chain,
        }

        if self.name is not None:
            fillet_json_name = {"name": self.name}
            fillet_json_name.update(fillet_json)
            fillet_json = fillet_json_name

        return fillet_json

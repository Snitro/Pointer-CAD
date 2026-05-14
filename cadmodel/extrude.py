from occwl.wire import Wire
from occwl.face import Face
from OCC.Core.gp import gp_Vec
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut

from misc import ROUND_JSON, token_quantize, token_dequantize, TOKEN
from .modelfeature import ModelFeature
from .sketch import Sketch


class Extrude(ModelFeature):
    def __init__(
        self,
        index,
        sketches: Sketch,
        extent_type,
        extent_one,
        extent_two,
        operation,
        bbox=None,
        name=None,
    ):
        """
        Args:
            sketches (Sketch): Sketch object containing profiles and sketch planes
            extent_type (str): EXTENT_TYPE
            extent_one (float): extrude distance in normal direction
            extent_two (float): extrude distance in opposite direction
            operation (int): EXTRUDE_OPERATIONS
        """
        super().__init__(index, bbox, name)

        self.sketches = sketches
        self.extent_type = extent_type
        self.extent_one = extent_one
        self.extent_two = extent_two
        self.operation = operation

    @staticmethod
    def start_token():
        return [TOKEN.index("<|extrude_start|>"), None]

    @staticmethod
    def from_dict(all_stat, extrude_id, index):
        """Construct an ``Extrude`` instance from JSON data.

        Args:
            all_stat (dict): Parsed CAD JSON data.
            extrude_id (str): Entity ID for this extrude feature.
            index (int): Feature index in the design sequence.

        Returns:
            Extrude: Decoded extrude feature.
        """
        extrude_entity = all_stat["features"][extrude_id]
        assert extrude_entity["type"] == "ExtrudeFeature"
        assert extrude_entity["start_extent"]["type"] == "ProfilePlaneStartDefinition"

        profiles = [
            (profile["feature"], profile["profile"])
            for profile in extrude_entity["profiles"]
        ]
        sketches = Sketch.from_dict(all_stat, profiles)

        extent_type = extrude_entity["extent_type"]

        extent_one = extrude_entity["extent_one"]["distance"]["value"]
        extent_two = 0.0
        if extent_type == "TwoSidesFeatureExtentType":
            extent_two = extrude_entity["extent_two"]["distance"]["value"]
        elif extent_type == "SymmetricFeatureExtentType":
            extent_two = extent_one

        if extent_one <= 0 and extent_two <= 0:
            extent_one, extent_two = -extent_two, -extent_one
        elif extent_one < 0 or extent_two < 0:
            raise ValueError("extrude distance must be positive")

        bbox = ModelFeature.parse_bounding_box(extrude_entity)

        operation = extrude_entity["operation"]
        name = extrude_entity["name"]

        return Extrude(
            index, sketches, extent_type, extent_one, extent_two, operation, bbox, name
        )

    def transform(self, movement=(0, 0, 0), scale=1):
        super().transform(movement, scale)

        self.sketches.transform(movement, scale)
        self.extent_one *= scale
        self.extent_two *= scale

    def build_part(self, pre_model=None):
        wires = self.sketches.build_wire()

        shapes = []
        for sketch_name, sketch_wires in wires.items():
            if len(sketch_wires) == 0:
                continue

            faces = self._make_face_from_wires(sketch_wires)
            if len(faces) == 0 or faces[0] is None:
                continue

            shape = self._extrude_from_face(faces[0])
            for i in range(1, len(faces)):
                inner_shape = self._extrude_from_face(faces[i])
                shape_cut = BRepAlgoAPI_Cut(shape, inner_shape)
                if not shape_cut.IsDone():
                    raise RuntimeError("Boolean union failed")
                shape = shape_cut.Shape()

            if shape is not None:
                shapes.append(shape)

        if len(shapes) == 0:
            return None

        finall_shape = shapes[0]
        for i in range(1, len(shapes)):
            finall_shape = BRepAlgoAPI_Fuse(finall_shape, shapes[i])
            if not finall_shape.IsDone():
                raise RuntimeError("Boolean union failed")
            finall_shape = finall_shape.Shape()

        return finall_shape

    def to_vector(self, built_solid=None, quant_bits=0):
        from misc import TOKEN

        boolean_map = {
            "NewBodyFeatureOperation": "<|extrude_new|>",
            "JoinFeatureOperation": "<|extrude_join|>",
            "CutFeatureOperation": "<|extrude_cut|>",
            "IntersectFeatureOperation": "<|extrude_intersect|>",
        }

        vec = self.sketches.to_vector(built_solid, quant_bits)

        ep = token_quantize(self.extent_one, 0, 2, quant_bits)
        en = token_quantize(self.extent_two, 0, 2, quant_bits)
        b = TOKEN.index(boolean_map[self.operation])
        vec.extend([Extrude.start_token(), [ep, None], [en, None], [b, None]])

        return vec

    @staticmethod
    def from_vector(vector, quant_bits=0, strict=True):
        boolean_map = {
            "NewBodyFeatureOperation": "<|extrude_new|>",
            "JoinFeatureOperation": "<|extrude_join|>",
            "CutFeatureOperation": "<|extrude_cut|>",
            "IntersectFeatureOperation": "<|extrude_intersect|>",
        }
        boolean_idx = {k: TOKEN.index(v) for k, v in boolean_map.items()}

        ST_positions = [
            i
            for i in reversed(range(len(vector)))
            if vector[i] == Extrude.start_token()
        ]

        if len(ST_positions) == 0:
            raise ValueError("Failed to parse vector: no extrude start token found.")

        for idx in ST_positions:
            try:
                ext_vector = vector[idx + 1 :]
                if len(ext_vector) < 3:
                    raise ValueError(
                        "Not enough elements to decode extrude from vector."
                    )

                OP_positions = [
                    i
                    for i in reversed(range(len(ext_vector)))
                    if ext_vector[i][0] in boolean_idx.values()
                ]
                if len(OP_positions) == 0:
                    raise ValueError(
                        "Not enough elements to decode extrude from vector."
                    )
                for op_idx in OP_positions:
                    try:
                        operation_idx = ext_vector[op_idx][0]
                        operation = list(boolean_idx.keys())[
                            list(boolean_idx.values()).index(operation_idx)
                        ]

                        extent_list = []
                        for extent_idx in range(op_idx):
                            if ext_vector[extent_idx][0] >= len(TOKEN):
                                extent_list.append(
                                    token_dequantize(
                                        ext_vector[extent_idx][0], 0, 2, quant_bits
                                    )
                                )
                        extent_one, extent_two = extent_list[:2]
                        if extent_one == extent_two == 0:
                            raise ValueError(
                                "Not enough elements to decode extrude from vector."
                            )
                        if extent_one == extent_two:
                            extent_type = "SymmetricFeatureExtentType"
                        elif extent_one > 0 and extent_two > 0:
                            extent_type = "TwoSidesFeatureExtentType"
                        else:
                            extent_type = "OneSideFeatureExtentType"

                        break
                    except Exception as e:
                        raise e
                else:
                    raise ValueError(
                        "Not enough elements to decode extrude from vector."
                    )

                skt_vector = vector[:idx]
                sketches = Sketch.from_vector(skt_vector, quant_bits, strict)
                break
            except Exception as e:
                raise e
        else:
            raise ValueError("Not enough elements to decode extrude from vector.")

        return Extrude(-1, sketches, extent_type, extent_one, extent_two, operation)

    def _make_face_from_wires(self, wires: list[Wire]):
        faces = []

        for wire in wires:
            face = None

            builder = BRepBuilderAPI_MakeFace(wire.topods_shape())
            builder.Build()
            if builder.IsDone():
                face = Face(builder.Face())

            faces.append(face)

        return faces

    def _extrude_from_face(self, face: Face):
        direction = self.sketches.ref_coordinate.normal

        prism_one, prism_two = None, None
        if self.extent_one > 0:
            prism_one = BRepPrimAPI_MakePrism(
                face.topods_shape(), gp_Vec(*(direction * self.extent_one)), True
            )
        if self.extent_two > 0:
            prism_two = BRepPrimAPI_MakePrism(
                face.topods_shape(), gp_Vec(*(-direction * self.extent_two)), True
            )

        if prism_one and not prism_one.IsDone():
            raise RuntimeError("prism_one failed")
        if prism_two and not prism_two.IsDone():
            raise RuntimeError("prism_two failed")

        shape_one = prism_one.Shape() if prism_one else None
        shape_two = prism_two.Shape() if prism_two else None

        if shape_one and shape_two:
            fused = BRepAlgoAPI_Fuse(shape_one, shape_two)
            if not fused.IsDone():
                raise RuntimeError("Boolean union failed")
            result = fused.Shape()

        elif shape_one:
            result = shape_one
        elif shape_two:
            result = shape_two
        else:
            result = None

        return result

    def _json(self):
        extent_one = self.extent_one
        extent_two = self.extent_two

        if self.extent_type == "SymmetricFeatureExtentType":
            extent_one *= 2
            extent_two = 0

        extrude_json = {
            "extrude_operation_type": self.operation,
            "extrude_extent_mode": self.extent_type,
            "extrude_depth_towards_normal": round(extent_one, ROUND_JSON),
            "extrude_depth_opposite_normal": round(extent_two, ROUND_JSON),
        }

        if self.name is not None:
            extrude_json_name = {"name": self.name}
            extrude_json_name.update(extrude_json)
            extrude_json = extrude_json_name

        return extrude_json

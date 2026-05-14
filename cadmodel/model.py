import dgl
import time
import copy
import torch
import numpy as np
from occwl.face import Face
from occwl.edge import Edge
from occwl.solid import Solid
from occwl.graph import face_adjacency
from occwl.uvgrid import ugrid, uvgrid
from occwl.entity_mapper import EntityMapper
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut, BRepAlgoAPI_Common

from misc import ROUND_JSON
from .fillet import Fillet
from .chamfer import Chamfer
from .extrude import Extrude
from .modelfeature import ModelFeature
from .occfix.unify import safe_unify_same_domain


def convert_json_from_deepcad(data):
    for seq_data in reversed(data["sequence"]):
        if seq_data["type"] == "ExtrudeFeature":
            feature_id = seq_data["entity"]
            if len(data["entities"][feature_id]["profiles"]) == 0:
                continue
            data["entities"][feature_id].update(data["properties"])
            break
    del data["properties"]

    for idx in range(len(data["sequence"])):
        data["sequence"][idx]["feature"] = data["sequence"][idx]["entity"]
        del data["sequence"][idx]["entity"]

    data["features"] = data["entities"]
    del data["entities"]

    for feature_name in data["features"]:
        if data["features"][feature_name]["type"] == "ExtrudeFeature":
            for idx in range(len(data["features"][feature_name]["profiles"])):
                data["features"][feature_name]["profiles"][idx]["feature"] = data[
                    "features"
                ][feature_name]["profiles"][idx]["sketch"]
                del data["features"][feature_name]["profiles"][idx]["sketch"]

    return data


class CADModel(object):
    def __init__(self, seq: list[ModelFeature] = None):
        self.seq = seq if seq is not None else []
        self._bbox_cache = (
            self.seq[-1].bbox.copy()
            if len(self.seq) > 0 and self.seq[-1].bbox is not None
            else None
        )  # 2 * 3 np array, (max, min) point

        self.movement = None
        self.scale = None
        self.occ_model = None

    @property
    def bbox(self):
        if self._bbox_cache is None:
            if self.occ_model is None:
                self.build_model()
            self._bbox_cache = np.array(
                [self.occ_model.box().max_point(), self.occ_model.box().min_point()]
            )

        return self._bbox_cache

    @bbox.setter
    def bbox(self, new_bbox):
        self._bbox_cache = new_bbox

    @staticmethod
    def from_dict(all_stat):
        """Construct a ``CADModel`` sequence from JSON data."""
        seq = []
        for item in all_stat["sequence"]:
            if item["type"] == "ExtrudeFeature":
                extrude_ops = Extrude.from_dict(
                    all_stat, item["feature"], item["index"]
                )
                if len(extrude_ops.sketches.profile_data) == 0:
                    continue
                seq.append(extrude_ops)
            elif item["type"] == "FilletFeature":
                fillet_ops = Fillet.from_dict(all_stat, item["feature"], item["index"])
                if len(fillet_ops.edges) == 0:
                    continue
                seq.append(fillet_ops)
            elif item["type"] == "ChamferFeature":
                chamfer_ops = Chamfer.from_dict(
                    all_stat, item["feature"], item["index"]
                )
                if len(chamfer_ops.edges) == 0:
                    continue
                seq.append(chamfer_ops)

        return CADModel(seq)

    def normalize(self, keep_sketch_plane=True, normalize_sketch=True):
        if not keep_sketch_plane:
            self.bbox = None

        bbox = self.bbox
        movement = -np.mean(bbox, axis=0)

        if keep_sketch_plane:
            min_bbox = np.min(bbox, axis=0)
            max_bbox = np.max(bbox, axis=0)
            crosses_plane = (min_bbox * max_bbox) <= 0
            movement *= (~crosses_plane).astype(int)

        self.movement = -movement
        for i in range(2):
            bbox[i] += movement

        scale = 1 / np.max(np.abs(bbox))
        self.scale = (1 / scale) if self.scale is None else (self.scale / scale)
        for i in range(2):
            bbox[i] *= scale

        self.bbox = bbox

        for operation in self.seq:
            operation.transform(movement=movement, scale=scale)
            if isinstance(operation, Extrude) and normalize_sketch:
                operation.sketches.normalize()

        self.occ_model = None

    def build_model(self, timeout: float = 300.0):
        if self.occ_model is not None:
            return self.occ_model

        start_time = time.time()
        model_shape = None

        for operation in self.seq:
            if isinstance(operation, Extrude):
                shape = operation.build_part(model_shape)
                if model_shape is None:
                    model_shape = shape
                else:
                    model_shape = self._merge_shape(
                        model_shape, shape, operation.operation
                    )
            else:
                if model_shape is None:
                    raise RuntimeError("Cannot perform operation without a base shape.")

                model_shape = operation.build_part(model_shape)

            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                model_shape = None
            else:
                model_shape = safe_unify_same_domain(
                    model_shape, timeout=remaining_time
                )
            if model_shape is None:
                break

        self.occ_model = (
            Solid(model_shape, allow_compound=True) if model_shape is not None else None
        )
        return self.occ_model

    def _merge_shape(self, base_shape, operate_shape, operation):
        if operation == "NewBodyFeatureOperation":
            builder = BRep_Builder()
            compound_combined = TopoDS_Compound()
            builder.MakeCompound(compound_combined)

            builder.Add(compound_combined, base_shape)
            builder.Add(compound_combined, operate_shape)

            return compound_combined
        else:
            merge_shape = None
            if operation == "JoinFeatureOperation":
                merge_shape = BRepAlgoAPI_Fuse(base_shape, operate_shape)
            elif operation == "CutFeatureOperation":
                merge_shape = BRepAlgoAPI_Cut(base_shape, operate_shape)
            elif operation == "IntersectFeatureOperation":
                merge_shape = BRepAlgoAPI_Common(base_shape, operate_shape)
            else:
                raise ValueError(f"Unsupported operation type: {operation}")

            if not merge_shape.IsDone():
                raise RuntimeError(
                    "Shape merging operation failed. The operation did not complete successfully."
                )

            return merge_shape.Shape()

    def submodel(self, idx):
        if idx < 0:
            return None

        return CADModel(copy.deepcopy(self.seq[: idx + 1]))

    def _map_entities_to_indices(self, data, solid: Solid = None, edges: list = None):
        """
        Recursively maps Faces and Edges in the given data structure to their corresponding indices.
        Faces are mapped to single integers.
        Edges are mapped to tuples of face indices, sorted in ascending order.

        Args:
            solid (Solid): The 3D solid object.
            data: A nested structure (e.g., list, Face, Edge, or other types).

        Returns:
            The data structure with Face and Edge entities replaced by their index representations.
        """
        from misc import STANDARD_PLANES

        mapper = EntityMapper(solid) if solid is not None else None

        def map_item(item):
            if isinstance(item, list):
                return [map_item(subitem) for subitem in item]

            if item is None:
                return -len(STANDARD_PLANES) - 1

            if isinstance(item, str):
                return list(STANDARD_PLANES.keys()).index(item) - len(STANDARD_PLANES)

            if mapper is not None and isinstance(item, Face):
                return mapper.face_index(item)

            if mapper is not None and edges is not None and isinstance(item, Edge):
                connected_faces = list(solid.faces_from_edge(item))
                face_indices = [mapper.face_index(face) for face in connected_faces]
                face_indices = tuple(sorted(face_indices))

                same_start_edges = set(solid.edges_from_face(face=connected_faces[0]))
                same_end_edges = set(solid.edges_from_face(face=connected_faces[1]))
                if len(same_start_edges & same_end_edges) != 1:
                    raise RuntimeError(
                        f"Edge connected to faces {face_indices} is not uniquely defined by its connected faces."
                    )

                edges_idx = edges.index(face_indices)
                return edges_idx

            return item  # Return as-is for other types

        return map_item(data)

    def to_vector(
        self, idx, curv_u_samples, surf_u_samples, surf_v_samples, quant_bits=0
    ):
        from misc import TOKEN

        if idx < 0 or idx >= len(self.seq):
            idx = len(self.seq) - 1

        pre_model = self.submodel(idx - 1)
        operation = self.seq[idx]

        built_solid = pre_model.build_model() if pre_model else None

        graph = None
        vec: list = operation.to_vector(built_solid, quant_bits)

        if idx == len(self.seq) - 1:
            vec.append([TOKEN.index("<|model_end|>"), None])
        else:
            vec.append([TOKEN.index("<|part_end|>"), None])

        if built_solid is not None:
            adjacency = face_adjacency(built_solid).to_undirected(as_view=True)

            # Compute the UV-grids for faces
            graph_face_feat = []
            for idx, face_idx in enumerate(adjacency.nodes):
                if idx != face_idx:
                    raise ValueError(
                        f"Face ID {face_idx} is not continuous. Unable to construct the graph."
                    )
                # Get the B-rep face
                face = adjacency.nodes[face_idx]["face"]
                # Compute UV-grids
                points = uvgrid(
                    face, method="point", num_u=surf_u_samples, num_v=surf_v_samples
                )
                normals = uvgrid(
                    face, method="normal", num_u=surf_u_samples, num_v=surf_v_samples
                )
                curvatures = uvgrid(
                    face,
                    method="gaussian_curvature",
                    num_u=surf_u_samples,
                    num_v=surf_v_samples,
                )
                visibility_status = uvgrid(
                    face,
                    method="visibility_status",
                    num_u=surf_u_samples,
                    num_v=surf_v_samples,
                )
                mask = np.logical_or(
                    visibility_status == 0, visibility_status == 2
                )  # 0: Inside, 1: Outside, 2: On boundary
                # Concatenate channel-wise to form face feature tensor
                face_feat = np.concatenate((points, normals, curvatures, mask), axis=-1)
                graph_face_feat.append(face_feat)
            graph_face_feat = np.asarray(graph_face_feat)

            # Compute the U-grids for edges
            graph_edge_feat = []
            for edge_idx in adjacency.edges:
                # Get the B-rep edge
                edge: Edge = adjacency.edges[edge_idx]["edge"]
                # Ignore degenerate edges, e.g. at apex of cone.
                if not edge.has_curve():
                    raise RuntimeError(
                        f"Unable to construct edge feature for edge ID {edge_idx}. The edge does not have a valid curve."
                    )
                # Compute U-grids
                points = ugrid(edge, method="point", num_u=curv_u_samples)
                tangents = ugrid(edge, method="tangent", num_u=curv_u_samples)
                derivatives = ugrid(
                    edge, method="first_derivative", num_u=curv_u_samples
                )
                # Concatenate channel-wise to form edge feature tensor
                edge_feat = np.concatenate(
                    (points, tangents, -tangents, derivatives), axis=-1
                )
                graph_edge_feat.append(edge_feat)
            graph_edge_feat = np.asarray(graph_edge_feat)

            # Convert face-adj graph to DGL format
            edges = list(adjacency.edges)
            src = [e[0] for e in edges]
            dst = [e[1] for e in edges]
            graph = dgl.graph((src, dst), num_nodes=len(adjacency.nodes))
            graph.ndata["x"] = torch.from_numpy(graph_face_feat)
            graph.edata["x"] = torch.from_numpy(graph_edge_feat)
            graph = dgl.add_reverse_edges(graph, copy_ndata=True, copy_edata=True)

            vec = self._map_entities_to_indices(vec, built_solid, list(zip(src, dst)))
        else:
            vec = self._map_entities_to_indices(vec)

        return graph, vec, float(self.scale), self.movement.tolist()

    def _map_entities_from_indices(self, vector: list):
        from misc import STANDARD_PLANES

        solid: Solid = self.build_model()

        faces = None
        edges = None
        if solid is not None:
            mapper = EntityMapper(solid)
            adjacency = face_adjacency(solid).to_undirected(as_view=True)

            faces = [None] * len(list(solid.faces()))
            for face in solid.faces():
                faces[mapper.face_index(face)] = face

            edges = [None] * len(list(solid.edges()))
            adj_edges = list(adjacency.edges)
            src = [e[0] for e in adj_edges]
            dst = [e[1] for e in adj_edges]
            adj_edges = list(zip(src, dst))
            for edge in solid.edges():
                connected_faces = list(solid.faces_from_edge(edge))
                face_indices = tuple(
                    sorted([mapper.face_index(face) for face in connected_faces])
                )
                if face_indices in adj_edges:
                    edges_idx = adj_edges.index(face_indices)
                    edges[edges_idx] = edge

        def map_item(item):
            if item < 0 and item >= -len(STANDARD_PLANES):
                return (list(STANDARD_PLANES.keys())[item + len(STANDARD_PLANES)], None)

            if item >= 0:
                face = faces[item] if item < len(faces) else None
                edge = edges[item] if item < len(edges) else None

                return (face, edge)

            return None

        return [[item[0], map_item(item[1])] for item in vector]

    def from_vector(self, vector, quant_bits=0, strict=True):
        from misc import TOKEN

        assert self.movement == self.scale == None

        vector = self._map_entities_from_indices(vector)
        ET_positions = [
            i
            for i in reversed(range(len(vector)))
            if (
                vector[i][0] == TOKEN.index("<|model_end|>")
                or vector[i][0] == TOKEN.index("<|part_end|>")
            )
        ]

        if len(ET_positions) == 0:
            raise ValueError(
                "Failed to parse vector: no model end or part end token found."
            )

        error = None
        for idx in ET_positions:
            design_vector = vector[:idx]
            extrude_start, chamfer_start, fillet_start = False, False, False

            for v in design_vector:
                if v == Extrude.start_token():
                    extrude_start = True
                elif v == Chamfer.start_token():
                    chamfer_start = True
                elif v == Fillet.start_token():
                    fillet_start = True

            if extrude_start:
                try:
                    operation: Extrude = Extrude.from_vector(
                        design_vector, quant_bits, strict
                    )
                    operation.index = len(self.seq)
                    break
                except Exception as e:
                    error = e

            if chamfer_start:
                try:
                    operation: Chamfer = Chamfer.from_vector(
                        design_vector, quant_bits, strict
                    )
                    operation.index = len(self.seq)
                    break
                except Exception as e:
                    error = e

            if fillet_start:
                try:
                    operation: Fillet = Fillet.from_vector(
                        design_vector, quant_bits, strict
                    )
                    operation.index = len(self.seq)
                    break
                except Exception as e:
                    error = e
        else:
            if error is not None:
                raise error
            else:
                raise ValueError("No valid operator found in the vector.")

        self.seq.append(operation)
        self._bbox_cache = self.occ_model = None

        return vector[idx][0] == TOKEN.index("<|model_end|>")

    def _json(self):
        """
        Convert the CADSequence object to a JSON object.

        Returns:
            dict: The JSON object.

        """
        cad_seq_repr = {}
        cad_seq_repr["parts"] = {}
        cad_seq_repr["dimensions"] = {}
        cad_seq_repr["caption"] = ""
        cad_seq_repr["visualization"] = ""

        x, y, z = np.abs(self.bbox[0] - self.bbox[1])
        cad_seq_repr["dimensions"]["x_length"] = round(x, ROUND_JSON)
        cad_seq_repr["dimensions"]["y_length"] = round(y, ROUND_JSON)
        cad_seq_repr["dimensions"]["z_length"] = round(z, ROUND_JSON)

        for i, operation in enumerate(self.seq):
            if isinstance(operation, Extrude):
                length, width, extra_height = operation.sketches.dimension
                height = operation.extent_one + operation.extent_two + extra_height

                cad_seq_repr["parts"][f"part_{i+1}"] = {}
                cad_seq_repr["parts"][f"part_{i+1}"][
                    f"sketches"
                ] = operation.sketches._json()
                cad_seq_repr["parts"][f"part_{i+1}"][f"extrusion"] = operation._json()
                cad_seq_repr["parts"][f"part_{i+1}"]["description"] = {
                    "caption": f"<id>{operation.index}</id>",
                    "visualization": "",
                    "length": round(length, ROUND_JSON),
                    "width": round(width, ROUND_JSON),
                    "height": round(height, ROUND_JSON),
                }
            elif isinstance(operation, Fillet):
                cad_seq_repr["parts"][f"part_{i+1}"] = {}
                cad_seq_repr["parts"][f"part_{i+1}"][f"fillet"] = operation._json()
                cad_seq_repr["parts"][f"part_{i+1}"]["description"] = {
                    "caption": f"<id>{operation.index}</id>",
                    "visualization": "",
                }
            elif isinstance(operation, Chamfer):
                cad_seq_repr["parts"][f"part_{i+1}"] = {}
                cad_seq_repr["parts"][f"part_{i+1}"][f"chamfer"] = operation._json()
                cad_seq_repr["parts"][f"part_{i+1}"]["description"] = {
                    "caption": f"<id>{operation.index}</id>",
                    "visualization": "",
                }
            else:
                raise RuntimeError(
                    "Unsupported operation type encountered in CADModel._json()."
                )

        return cad_seq_repr

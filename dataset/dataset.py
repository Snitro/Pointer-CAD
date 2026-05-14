import torch
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import json
import os
import dgl
import pickle
from loguru import logger
from itertools import product
import torch.distributed as dist
from dgl.data.utils import load_graphs
from .augmentation import randomly_enhance_cad_data, format_cad_data
from misc import STANDARD_PLANES, TOKEN
from cadmodel.model import convert_json_from_deepcad


class GroupedDistributedSampler(DistributedSampler):
    def __init__(self, dataset, group_size=1, **kwargs):
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        assert (
            world_size % group_size == 0
        ), "World size must be divisible by group_size"

        # Map rank to a shared virtual rank within each group.
        group_id = rank // group_size
        fake_rank = group_id
        fake_world_size = world_size // group_size

        # Build the base class with group-level rank/world_size.
        super().__init__(
            dataset, num_replicas=fake_world_size, rank=fake_rank, **kwargs
        )

        # Keep real rank/group info for possible downstream use.
        self.real_rank = rank
        self.group_id = group_id
        self.group_size = group_size


class PointerCADDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        split_filepath: str,
        subset: str,
        augment: bool = False,
        prompt_choices=["abs", "exp"],
        max_graph_edge_num=1000,
        max_graph_node_num=200,
    ):
        """
        Args:
            dataset_dir (str): Dataset root directory.
            split_filepath (str): Train/validation/test split JSON path.
            subset (str): One of "train", "test", or "validation".
            augment (bool): Whether to apply data augmentation.
            prompt_choices (list[str]): Prompt template types to sample.
            max_graph_edge_num (int): Max edge count after graph clipping.
            max_graph_node_num (int): Max node count after graph clipping.
        """
        super(PointerCADDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.augment = augment
        self.prompt_choices = prompt_choices
        self.max_graph_edge_num = max_graph_edge_num
        self.max_graph_node_num = max_graph_node_num

        # Load split JSON file.
        with open(os.path.join(split_filepath), "r") as f:
            self.split = json.load(f)

        self.data_id = self.split[subset]
        if not augment:
            self.data_id = sorted(self.data_id)

        logger.info(f"Found {len(self)} samples for {subset} split.")

    def __len__(self):
        return len(self.data_id) * len(self.prompt_choices)

    def _flatten_vector(self, data):
        if data[0][0] == TOKEN.index("<|sketch_start|>"):
            values, pointers = [[[]]], [[[]]]
            for pair in data:
                if (
                    len(pair) == 2
                    and not isinstance(pair[0], list)
                    and not isinstance(pair[1], list)
                ):
                    values[-1][0].append(pair[0])
                    pointers[-1][0].append(pair[1])
                elif (
                    len(pair) == 2
                    and not isinstance(pair[0], list)
                    and isinstance(pair[1], list)
                ):
                    values[-1][0].append(pair[0])
                    pointers.append([[v] for v in pair[1]])
                    pointers.append([[]])
                else:
                    value = []
                    for vector_list in pair:
                        value.append([a for a, _ in vector_list])
                    values.append(value)
                    values.append([[]])
                    pointers[-1][0].extend([pair[0][0][1]] * len(pair[0]))

            value = [item for sublist in values for item in random.choice(sublist)]
            value_source = [
                [item for sublist in v for item in sublist] for v in product(*values)
            ]
            pointer = [item for sublist in pointers for item in random.choice(sublist)]
            pointer_source = []
            for sublist in pointers:
                if sublist[0][0] >= -len(STANDARD_PLANES):
                    pointer_source.append([item[0] for item in sublist])
                else:
                    assert len(sublist) == 1
                    for item in sublist[0]:
                        assert item == -len(STANDARD_PLANES) - 1
                        pointer_source.append(item)
        elif data[0][0] in [
            TOKEN.index("<|chamfer_start|>"),
            TOKEN.index("<|fillet_start|>"),
        ]:
            assert len(data) == 4, "Chamfer and fillet vectors should have 4 items."
            pointers = list(set(data[2][1]))
            extra_item_num = len(pointers)

            value = [data[0][0], data[1][0]]
            value.extend([data[2][0] for _ in range(extra_item_num)])
            value.append(data[3][0])

            pointer = [data[0][1], data[1][1]]
            shuffled_items = pointers[:]
            random.shuffle(shuffled_items)
            pointer.extend(shuffled_items)
            pointer.append(data[3][1])

            value_source = [value[:]]
            pointer_source = pointer.copy()
            pointer_source[2] = pointers[:]
            for idx in range(3, len(pointer_source) - 1):
                pointer_source[idx] = []

        return value, value_source, pointer, pointer_source

    def _clip_graph(self, g: dgl.DGLGraph, keep_nodes, keep_edges):
        """
        Clip graph size while preserving required nodes and edges.

        Args:
            g (dgl.DGLGraph): Input graph.
            keep_nodes (list[int]): Node IDs that must be preserved.
            keep_edges (list[int]): Edge IDs that must be preserved.
        """
        remove_eids = list(range(g.num_edges() // 2, g.num_edges()))
        g_pruned = dgl.remove_edges(g, remove_eids)

        all_edges = set(range(g_pruned.num_edges()))
        selected_edges = set(keep_edges)

        unused_nodes = set(keep_nodes)
        for eid in range(g_pruned.num_edges()):
            src, dst = g_pruned.find_edges(eid)
            if src.item() in unused_nodes and dst.item() in unused_nodes:
                selected_edges.add(eid)
                unused_nodes.remove(src.item())
                unused_nodes.remove(dst.item())
            if len(unused_nodes) == 0:
                break

        for nid in unused_nodes:
            in_eids = g_pruned.in_edges(nid, form="eid")
            out_eids = g_pruned.out_edges(nid, form="eid")
            if len(in_eids) > 0:
                selected_edges.add(in_eids.tolist()[0])
            else:
                selected_edges.add(out_eids.tolist()[0])

        if len(selected_edges) < self.max_graph_edge_num // 2:
            remaining = list(all_edges - selected_edges)
            extra = remaining[: self.max_graph_edge_num // 2 - len(selected_edges)]
            selected_edges.update(extra)

        sg: dgl.DGLGraph = g_pruned.edge_subgraph(
            list(selected_edges), relabel_nodes=True, store_ids=True
        )

        new_nid: list = sg.ndata[dgl.NID].tolist()
        new_eid: list = sg.edata[dgl.EID].tolist()

        node_map = {
            keep_node: new_nid.index(keep_node)
            for keep_node in keep_nodes
            if keep_node in new_nid
        }
        edge_map = {
            keep_edge: new_eid.index(keep_edge)
            for keep_edge in keep_edges
            if keep_edge in new_eid
        }

        if sg.num_nodes() <= self.max_graph_node_num:
            graph = dgl.add_reverse_edges(sg, copy_ndata=True, copy_edata=True)
            return graph, node_map, edge_map

        selected_nodes = set(node_map.values())
        for eid in list(edge_map.values()) if len(edge_map) > 0 else [0]:
            src, dst = sg.find_edges(eid)
            selected_nodes.add(src.item())
            selected_nodes.add(dst.item())

        if len(selected_nodes) < self.max_graph_node_num:
            remaining = list(set(range(sg.num_nodes())) - selected_nodes)
            extra = remaining[: self.max_graph_node_num - len(selected_nodes)]
            selected_nodes.update(extra)

        ssg = sg.subgraph(list(selected_nodes), relabel_nodes=True, store_ids=True)
        ssg_new_nid: list = ssg.ndata[dgl.NID].tolist()
        ssg_new_eid: list = ssg.edata[dgl.EID].tolist()

        ssg_node_map = {
            keep_node: ssg_new_nid.index(keep_node_sg)
            for keep_node, keep_node_sg in node_map.items()
            if keep_node_sg in ssg_new_nid
        }
        ssg_edge_map = {
            keep_edge: ssg_new_eid.index(keep_edge_sg)
            for keep_edge, keep_edge_sg in edge_map.items()
            if keep_edge_sg in ssg_new_eid
        }

        graph = dgl.add_reverse_edges(ssg, copy_ndata=True, copy_edata=True)
        return graph, ssg_node_map, ssg_edge_map

    def _prepare_data(self, chunk, model_id, part_id, prompt_choice):
        prompt_path = os.path.join(
            self.dataset_dir, chunk, model_id, f"prompt_{prompt_choice}.txt"
        )
        with open(prompt_path, "r", encoding="utf-8") as file:
            prompt = file.read()

        vec_path = os.path.join(
            self.dataset_dir, chunk, model_id, "vector", f"{model_id}_{part_id}.pkl"
        )
        with open(vec_path, "rb") as f:
            vector_dict = pickle.load(f)

            vector = self._flatten_vector(vector_dict["vec"])
            scale = vector_dict["scale"]
            movement = vector_dict["movement"]

            vector_value = torch.tensor(vector[0])
            vector_value_source = torch.tensor(vector[1])
            vector_pointer = torch.tensor(vector[2])
            vector_pointer_source = vector[3]
            assert vector_value.shape == vector_pointer.shape

        graph_path = os.path.join(
            self.dataset_dir, chunk, model_id, "graph", f"{model_id}_{part_id}.bin"
        )
        if os.path.exists(graph_path):
            graph = load_graphs(graph_path)[0][0]
            graph.ndata["x"] = graph.ndata["x"].type(torch.float32)
            graph.edata["x"] = graph.edata["x"].type(torch.float32)
            if len(graph.ndata["x"].shape) == 4:
                graph.ndata["x"][:, :, :, -2] = torch.clamp(
                    graph.ndata["x"][:, :, :, -2], min=-10, max=10
                )
            if len(graph.edata["x"].shape) == 3:
                graph.edata["x"][:, :, -3:] = torch.clamp(
                    graph.edata["x"][:, :, -3:], min=-100, max=100
                )

            if (
                graph.num_edges() > self.max_graph_edge_num
                or graph.num_nodes() > self.max_graph_node_num
            ):
                used_face, used_edge = [], []
                for v, p in zip(vector_value, vector_pointer_source[1:]):
                    if isinstance(p, list):
                        if v == TOKEN.index("<|sketch_start|>"):
                            used_face.extend([x for x in p if x >= 0])
                        else:
                            used_edge.extend(p)

                graph, graph_face_map, graph_edge_map = self._clip_graph(
                    graph, used_face, used_edge
                )

                for idx in range(1, len(vector_pointer_source)):
                    if isinstance(vector_pointer_source[idx], list):
                        if vector_value[idx - 1] == TOKEN.index("<|sketch_start|>"):
                            pointer_map = graph_face_map
                        else:
                            pointer_map = graph_edge_map

                        if vector_pointer[idx].item() >= 0:
                            vector_pointer[idx] = pointer_map[
                                vector_pointer[idx].item()
                            ]
                        vector_pointer_source[idx] = [
                            pointer_map[x] if x >= 0 else x
                            for x in vector_pointer_source[idx]
                        ]

            if "_ID" in graph.ndata:
                del graph.ndata["_ID"]
            if "_ID" in graph.edata:
                del graph.edata["_ID"]
        else:
            graph = dgl.graph(([], []))
            graph.ndata["x"] = torch.zeros((0,), dtype=torch.float32)
            graph.edata["x"] = torch.zeros((0,), dtype=torch.float32)

        json_path = os.path.join(
            self.dataset_dir, chunk, model_id, "json", f"{model_id}_{part_id}.json"
        )
        if os.path.exists(json_path):
            with open(json_path, "r") as fp:
                json_dict = json.load(fp)
                if "properties" in json_dict:
                    json_dict = convert_json_from_deepcad(json_dict)
        else:
            json_path = os.path.join(
                self.dataset_dir, chunk, model_id, f"{model_id}.json"
            )
            with open(json_path, "r") as fp:
                json_dict = json.load(fp)
                if "properties" in json_dict:
                    json_dict = convert_json_from_deepcad(json_dict)
            sequence = json_dict["sequence"]
            json_dict["sequence"] = []
            for item in sequence:
                if "index" in item and item["index"] <= int(part_id):
                    json_dict["sequence"].append(item)

        return (
            chunk,
            model_id,
            part_id,
            vector_value,
            vector_value_source,
            vector_pointer,
            vector_pointer_source,
            prompt,
            scale,
            movement,
            graph,
            json_dict,
        )

    def __getitem__(self, idx):
        prompt_choice = self.prompt_choices[int(idx / len(self.data_id))]
        chunk, model_id, part_id = self.data_id[idx % len(self.data_id)].split("_")

        data = self._prepare_data(chunk, model_id, part_id, prompt_choice)
        (
            chunk,
            model_id,
            part_id,
            vector_value,
            vector_value_source,
            vector_pointer,
            vector_pointer_source,
            prompt,
            scale,
            movement,
            graph,
            json_dict,
        ) = data
        if self.augment:
            return (
                chunk,
                model_id,
                part_id,
                vector_value,
                vector_value_source,
                vector_pointer,
                vector_pointer_source,
                *randomly_enhance_cad_data(prompt, scale, movement),
                graph,
                json_dict,
            )
        else:
            return (
                chunk,
                model_id,
                part_id,
                vector_value,
                vector_value_source,
                vector_pointer,
                vector_pointer_source,
                *format_cad_data(prompt, scale, movement),
                graph,
                json_dict,
            )


def collate(batch):
    chunks = []
    model_ids = []
    part_ids = []
    values = []
    value_sources = []
    pointers = []
    pointer_sources = []
    prompts = []
    scales = []
    movements = []
    graphs = []
    jsons = []

    for sample in batch:
        (
            chunk,
            model_id,
            part_id,
            vector_value,
            vector_value_source,
            vector_pointer,
            vector_pointer_source,
            prompt,
            scale,
            movement,
            graph,
            json_dict,
        ) = sample

        chunks.append(chunk)
        model_ids.append(model_id)
        part_ids.append(part_id)
        values.append(vector_value)
        value_sources.append(vector_value_source)
        pointers.append(vector_pointer)
        pointer_sources.append(vector_pointer_source)
        prompts.append(prompt)
        scales.append(scale)
        movements.append(movement)
        graphs.append(graph)
        jsons.append(json_dict)

    return {
        "chunk": chunks,  # list of strings
        "model_id": model_ids,  # list of strings
        "part_id": part_ids,  # list of strings
        "value": values,  # list of tensor
        "value_source": value_sources,  # list of tensor
        "pointer": pointers,  # list of tensor
        "pointer_source": pointer_sources,  # list of list
        "prompt": prompts,  # list of strings
        "scale": torch.tensor(scales, dtype=torch.float32),  # shape (B)
        "movement": torch.tensor(movements, dtype=torch.float32),  # shape (B, 3)
        "graph": dgl.batch(graphs),
        "json": jsons,
    }


def get_dataloaders(
    dataset_dir: str,
    split_filepath: str,
    subsets: list[str],
    batch_sizes,
    group_sizes=1,
    augment: bool = True,
    shuffle: bool = True,
    pin_memory: bool = False,
    num_workers: int = 4,
    prefetch_factor: int = 16,
    prompt_choices=["abs", "exp"],
):
    """
    Create one dataloader per requested subset.

    Args:
        dataset_dir (str): Dataset root directory.
        split_filepath (str): Split JSON path.
        subsets (list[str]): Subsets to load.
        batch_sizes (int | list[int]): Batch size(s) per subset.
        group_sizes (int | list[int]): Distributed group size(s) per subset.
        augment (bool): Whether to apply augmentation in dataset.
        shuffle (bool): Whether to shuffle each subset.
        pin_memory (bool): DataLoader pin_memory option.
        num_workers (int): Number of DataLoader workers.
        prefetch_factor (int): DataLoader prefetch factor.
        prompt_choices (list[str]): Prompt templates to use.

    Returns:
        list[DataLoader]: One dataloader for each requested subset.
    """

    all_dataloaders = []

    if isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes] * len(subsets)
    if isinstance(group_sizes, int):
        group_sizes = [group_sizes] * len(subsets)

    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = None  # os.sched_getaffinity not available on this OS
    if cpu_count is not None and num_workers is not None and num_workers > cpu_count:
        logger.warning(
            f"num_workers ({num_workers}) is greater than the number of CPU cores ({cpu_count}). This may cause performance issues."
        )

    for subset, batch_size, group_size in zip(subsets, batch_sizes, group_sizes):
        dataset = PointerCADDataset(
            dataset_dir=dataset_dir,
            split_filepath=split_filepath,
            subset=subset,
            augment=augment,
            prompt_choices=prompt_choices,
        )

        train_sampler = (
            GroupedDistributedSampler(
                dataset, group_size=group_size, shuffle=shuffle, drop_last=True
            )
            if dist.is_initialized()
            else None
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False if dist.is_initialized() else shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,  # Set to True if using CUDA
            prefetch_factor=prefetch_factor,
            sampler=train_sampler,
            collate_fn=collate,
        )
        all_dataloaders.append(dataloader)

    return all_dataloaders

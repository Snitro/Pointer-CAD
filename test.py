import os
import gc
import dgl
import json
import yaml
import torch
import requests
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from loguru import logger
from requests.adapters import HTTPAdapter

from occwl.graph import face_adjacency
from occwl.uvgrid import ugrid, uvgrid

from measurements.intersection_over_union import intersection_over_union
from measurements.chamfer_distance import chamfer_distance
from measurements.f1_score import f1_score
from measurements.watertightness import is_watertight
from misc import MAX_PART_LENGTH, create_mesh
from cadmodel.model import CADModel
from models.pointercad import PointerCAD
from dataset.dataset import get_dataloaders
from models.processor import PointerCADProcessor


def parse_config_file(config_file):
    with open(config_file, "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data


def get_brep(model: CADModel, surf_u_samples=32, surf_v_samples=32, curv_u_samples=32):
    success = True

    if model is not None:
        try:
            adjacency = face_adjacency(model.build_model()).to_undirected(as_view=True)

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
                edge = adjacency.edges[edge_idx]["edge"]
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
            graph.ndata["x"] = torch.from_numpy(graph_face_feat).to(torch.float32)
            graph.edata["x"] = torch.from_numpy(graph_edge_feat).to(torch.float32)
            graph = dgl.add_reverse_edges(graph, copy_ndata=True, copy_edata=True)
            if "_ID" in graph.ndata:
                del graph.ndata["_ID"]
            if "_ID" in graph.edata:
                del graph.edata["_ID"]
            return graph, success
        except Exception as e:
            logger.warning(f"An error occurred while constructing the B-rep graph: {e}")
            success = False

    graph = dgl.graph(([], []))
    graph.ndata["x"] = torch.zeros((0,), dtype=torch.float32)
    graph.edata["x"] = torch.zeros((0,), dtype=torch.float32)
    return graph, success


def job_online(
    device_name, dataset_size, log_dir, config, host: str, session: requests.Session
):
    """
    Notify the server that the client is online and ready to process data.
    """
    url = f"http://{host}:32500/connect"
    response = session.post(
        url,
        json={
            "gpu_info": device_name,
            "dataset_size": dataset_size,
            "log_dir": log_dir,
            "config": config,
        },
    )
    return response.json()


def request_model_id(model_id, host: str, session: requests.Session):
    """
    Request a model ID from the server.
    """
    url = f"http://{host}:32500/apply_model_id"
    response = session.post(url, json={"model_id": model_id})
    return response.json()["allowed"]


def report_result(model_id, result, host: str, session: requests.Session):
    """
    Report the result of processing a model ID to the server.
    """
    url = f"http://{host}:32500/report"
    response = session.post(url, json={"model_id": model_id, "result": result})
    return response.json()


def ping(host: str, session: requests.Session):
    """
    Ping the server to check if it is alive.
    """
    url = f"http://{host}:32500/ping"
    response = session.get(url)
    return response.json()


def finish_test(device_name, host: str, session: requests.Session):
    """
    Notify the server that the client has finished processing.
    """
    url = f"http://{host}:32500/finish"
    response = session.post(url, json={"gpu_info": device_name})
    session.close()
    return response.json()


def create_session():
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=16,
        pool_maxsize=32,
        max_retries=3,
        pool_block=True,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


@logger.catch()
def main():
    # Use add_help=False to free up '-h' for host; provide '--help' for help output
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-c", "--config_path", type=str, default="./config/test.yaml")
    parser.add_argument(
        "-h",
        "--host",
        type=str,
        default="localhost",
        help="Server host name or IP (port 32500 is assumed). Default: localhost",
    )
    parser.add_argument(
        "--help", action="help", help="Show this help message and exit."
    )
    args = parser.parse_args()

    config = parse_config_file(args.config_path)
    device = torch.device("cuda")
    logger.info(f"Current Device {torch.cuda.get_device_properties(device)}")

    pointercad = PointerCAD(
        qwen_model=config["model"]["base_model"], bit=config["model"]["bit"]
    ).to(device)

    pointercad.model.print_trainable_parameters()

    processor: PointerCADProcessor = PointerCADProcessor.from_pretrained(
        pretrained_model_name_or_path=config["model"]["base_model"], padding_side="left"
    )

    now = datetime.datetime.now()
    time_str = now.strftime("%H:%M")
    date_str = datetime.date.today()
    log_dir = os.path.join(config["test"]["log_dir"], f"{date_str}/{time_str}")

    logger.info(f"Current Date {date_str} Time {time_str}\n")

    test_model(
        model=pointercad,
        processor=processor,
        device=device,
        log_dir=log_dir,
        config=config,
        host=args.host,
    )


def test_model(model, processor, device, log_dir, config, host: str):
    """
    Evaluate the model on the test split and report results to the server.

    Parameters:
        model (torch.nn.Module): The neural network model.
        device (str): Device used for evaluation.
        log_dir (str): Directory to save logs and predictions.
        config (dict): Evaluation configuration.
        host (str): Server host name or IP (without port).
    """
    test_loader = get_dataloaders(
        dataset_dir=config["dataset"]["dataset_dir"],
        split_filepath=config["dataset"]["split_filepath"],
        subsets=["test"],
        augment=False,
        batch_sizes=1,
        num_workers=config["test"]["num_workers"],
        pin_memory=False,
        shuffle=True,
        prefetch_factor=config["test"]["prefetch_factor"],
        prompt_choices=config["dataset"]["prompt"],
    )[0]

    checkpoint_file = config["test"]["checkpoint_path"]
    if checkpoint_file is not None and os.path.exists(checkpoint_file):
        logger.info(f"Using saved checkpoint at {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=device)

        if "epoch" in checkpoint:
            logger.info(f"Model was trained for epoch {checkpoint['epoch']}.")

        missing_keys_info = model.load_state_dict(checkpoint["model"], strict=False)
        if len(missing_keys_info.missing_keys) > 0:
            logger.warning(
                f"Missing keys in the checkpoint: {[key.split('.')[0] for key in missing_keys_info.missing_keys]}"
            )
    else:
        logger.error(
            "No checkpoint specified or checkpoint file does not exist. Unable to load pretrained model."
        )
        raise RuntimeError(
            "Checkpoint not specified or not found. Cannot load pretrained model."
        )

    session = create_session()

    online_info = job_online(
        device_name=torch.cuda.get_device_properties(device).name,
        dataset_size=len(test_loader),
        log_dir=log_dir,
        config=config,
        host=host,
        session=session,
    )
    log_dir = online_info["log_dir"]

    batch_size = config["test"]["batch_size"]
    test_acc_uid = {}
    model.eval()
    topk = 1

    iou = None  # Intersection Over Union
    cd = None  # Chamfer Distance

    with torch.no_grad():
        with tqdm(
            total=len(test_loader), ascii=True, dynamic_ncols=True, desc=f"Test ✨"
        ) as pbar:
            outof_model = False
            test_iter = iter(test_loader)

            total_test = success_samples = 0
            batch_model_id = []
            gt_models: dict[str, CADModel] = {}
            pred_models: dict[str, CADModel] = {}
            pred_models_scale: dict[str, float] = {}
            messages: dict[str, dict] = {}
            results = {}
            while not outof_model or len(batch_model_id) > 0:
                try:
                    while len(batch_model_id) < batch_size:
                        iter_dict = next(test_iter)
                        model_id = iter_dict["model_id"][0]

                        if not request_model_id(model_id, host, session=session):
                            pbar.update(1)
                            continue

                        gt_cadmodel = CADModel.from_dict(iter_dict["json"][0])
                        gt_models[model_id] = gt_cadmodel
                        messages[model_id] = [
                            {
                                "role": "system",
                                "content": "You are an expert mechanical engineer. Based on the user's text requirements, generate the corresponding CAD model design.",
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "brep"},
                                    {"type": "text", "text": iter_dict["prompt"][0]},
                                ],
                            },
                        ]
                        batch_model_id.append(model_id)
                        total_test += 1
                except StopIteration:
                    outof_model = True

                if len(batch_model_id) > 0:
                    finished_model = {}
                    batch_breps = []
                    batch_messages = []
                    for key in batch_model_id:
                        brep, success = get_brep(
                            pred_models[key] if key in pred_models else None
                        )
                        if not success:
                            logger.warning(
                                f"Failed to construct the B-rep graph for model_id {key}. Skipping this sample."
                            )
                            finished_model[key] = "failed to construct B-rep"

                        batch_breps.append(brep)
                        batch_messages.append(messages[key])

                    text = processor.apply_chat_template(
                        batch_messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = processor(
                        text=text, breps=dgl.batch(batch_breps), max_length=config["dataset"]["max_seq_len"]
                    ).to(device)

                    generated_value, generated_pointer, generated_scale = model.predict(
                        **inputs
                    )

                    gc.collect()
                    torch.cuda.empty_cache()

                    for model_id, pred_value, pred_pointer, pred_scale in zip(
                        batch_model_id,
                        generated_value,
                        generated_pointer,
                        generated_scale,
                    ):
                        if model_id not in pred_models:
                            pred_models[model_id] = CADModel()
                        if model_id not in pred_models_scale:
                            pred_models_scale[model_id] = -1

                        if pred_scale > 0 and pred_models_scale[model_id] > 0:
                            pred_models_scale[model_id] = (
                                pred_models_scale[model_id] * 0.5 + pred_scale * 0.5
                            )
                        elif pred_scale > 0:
                            pred_models_scale[model_id] = pred_scale

                        assert pred_value.shape == pred_pointer.shape
                        pred_vector = [
                            [pred_value[idx].item(), pred_pointer[idx].item()]
                            for idx in range(pred_value.shape[0])
                        ]

                        if len(pred_vector) == 0:
                            logger.warning(
                                f"Predicted vector for model_id {model_id} is empty or exceeds the maximum allowed length. Skipping this sample."
                            )
                            finished_model[model_id] = "out of length"
                            continue

                        try:
                            if pred_models[model_id].from_vector(
                                pred_vector, config["model"]["bit"]
                            ):
                                finished_model[model_id] = None
                            elif len(pred_models[model_id].seq) >= MAX_PART_LENGTH:
                                logger.warning(
                                    f"Predicted sequence for model_id {model_id} exceeds the maximum allowed part length ({MAX_PART_LENGTH}). Skipping this sample."
                                )
                                finished_model[model_id] = (
                                    "exceeded maximum part length"
                                )
                        except Exception as e:
                            logger.warning(
                                f"Exception occurred while building CADModel from prediction for model_id {model_id}: {e}"
                            )
                            finished_model[model_id] = "failed while building CADModel"

                            print(pred_vector)

                    for model_id, error_reason in finished_model.items():
                        try:
                            if error_reason is not None:
                                results[model_id] = {
                                    "status": False,
                                    "error_message": error_reason,
                                }
                                continue

                            gt_model = gt_models[model_id].build_model()
                            pred_model = (
                                pred_models[model_id].build_model()
                                if gt_model is not None
                                else None
                            )

                            if gt_model is None or pred_model is None:
                                results[model_id] = {
                                    "status": False,
                                    "error_message": "failed to build the CADModel",
                                }
                                continue

                            try:
                                gt_models[model_id].normalize(keep_sketch_plane=False)
                                pred_models[model_id].normalize(keep_sketch_plane=False)

                                gt_normalized_model = gt_models[model_id].build_model()
                                pred_normalized_model = (
                                    pred_models[model_id].build_model()
                                    if gt_normalized_model is not None
                                    else None
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Exception occurred while normalizing CADModel for model_id {model_id}: {e}"
                                )
                                pred_normalized_model = None

                            if (
                                gt_normalized_model is None
                                or pred_normalized_model is None
                            ):
                                results[model_id] = {
                                    "status": False,
                                    "error_message": "failed to normalize the CADModel",
                                }
                                continue

                            try:
                                iou = (
                                    intersection_over_union(
                                        pred_models[model_id], gt_models[model_id]
                                    )
                                    * 100
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Error occurred while calculating IOU for model_id {model_id}: {e}"
                                )
                                iou = None

                            try:
                                cd = (
                                    chamfer_distance(
                                        pred_models[model_id], gt_models[model_id], 8192
                                    )
                                    * 1000
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Error occurred while calculating Chamfer Distance for model_id {model_id}: {e}"
                                )
                                cd = None

                            try:
                                f1 = f1_score(
                                    pred_models[model_id], gt_models[model_id]
                                )
                                f1 = {k: v * 100 for k, v in f1.items()}
                            except Exception as e:
                                logger.warning(
                                    f"Error occurred while calculating F1 Score for model_id {model_id}: {e}"
                                )
                                f1 = None

                            try:
                                watertightness = is_watertight(pred_models[model_id])
                            except Exception as e:
                                logger.warning(
                                    f"Error occurred while checking watertightness for model_id {model_id}: {e}"
                                )
                                watertightness = None

                            if config["test"]["save_stl"]:
                                # save predicted STL
                                save_dir = os.path.join(log_dir, "stl")
                                os.makedirs(save_dir, exist_ok=True)
                                try:
                                    mesh = create_mesh(pred_models[model_id])
                                    mesh.export(os.path.join(save_dir, f"{model_id}.stl"))
                                except Exception as e:
                                    logger.warning(
                                        f"Error occurred while saving predicted STL for model_id {model_id}: {e}"
                                    )

                            if iou is not None or cd is not None:
                                results[model_id] = {
                                    "status": True,
                                    "intersection over union": iou,
                                    "chamfer distance": cd,
                                    "f1": f1,
                                    "is watertight": watertightness,
                                }
                                success_samples += 1
                            else:
                                results[model_id] = {
                                    "status": False,
                                    "error_message": "failed to measure result",
                                }
                        except Exception as e:
                            logger.warning(
                                f"Error occurred while measuring result for model_id {model_id}: {e}"
                            )
                            results[model_id] = {
                                "status": False,
                                "error_message": "unknown error",
                            }

                    for model_id in finished_model:
                        report_result(
                            model_id, results[model_id], host, session=session
                        )

                    batch_model_id = [
                        item for item in batch_model_id if item not in finished_model
                    ]

                    pbar.set_postfix(
                        {
                            "IOU": f"{iou:.2f}" if iou is not None else "N/A",
                            "CD": f"{cd:.4f}" if cd is not None else "N/A",
                            "IR": (
                                f"{(total_test - success_samples - len(batch_model_id)) / total_test * 100:.2f}%"
                                if total_test > 0
                                else "N/A"
                            ),
                        }
                    )
                    pbar.update(len(finished_model))

                    ping(host, session=session)

    finish_test(
        device_name=torch.cuda.get_device_properties(device).name,
        host=host,
        session=session,
    )


if __name__ == "__main__":
    main()

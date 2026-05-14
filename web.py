import os
import re
import dgl
import yaml
import json
import torch
import getpass
import tempfile
import datetime
import argparse
import numpy as np
import gradio as gr
from loguru import logger

from occwl.graph import face_adjacency
from occwl.uvgrid import ugrid, uvgrid

from misc import create_mesh
from cadmodel.model import CADModel
from models.pointercad import PointerCAD
from models.processor import PointerCADProcessor


def load_model(config, device):
    logger.info("Loading model...", config["web"]["checkpoint_path"])
    pointercad = PointerCAD(qwen_model=config["model"]["base_model"]).to(device)
    checkpoint = torch.load(config["web"]["checkpoint_path"], map_location=device)
    missing_keys_info = pointercad.load_state_dict(checkpoint["model"], strict=False)
    if len(missing_keys_info.missing_keys) > 0:
        logger.warning(
            f"Missing keys in the checkpoint: {[key.split('.')[0] for key in missing_keys_info.missing_keys]}"
        )

    return pointercad


def get_brep(model: CADModel, surf_u_samples=32, surf_v_samples=32, curv_u_samples=32):
    if model is not None and len(model.seq) > 0:
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
            derivatives = ugrid(edge, method="first_derivative", num_u=curv_u_samples)
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
        return graph

    graph = dgl.graph(([], []))
    graph.ndata["x"] = torch.zeros((0,), dtype=torch.float32)
    graph.edata["x"] = torch.zeros((0,), dtype=torch.float32)
    return graph


def format_prompt(user_prompt, system_message="You are a CAD designer"):
    """
    Formats the input prompt to match the model's expected structure.

    Args:
        user_prompt (str): The user's message.
        system_message (str, optional): The system message to guide the assistant.

    Returns:
        str: Formatted prompt.
    """
    message = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": [
                {"type": "brep"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    text = processor.apply_chat_template(
        [message], tokenize=False, add_generation_prompt=True
    )

    return text


def test_model(model: PointerCAD, prompt, device, use_sampling):
    mesh = None
    pred_model = CADModel()

    with torch.no_grad():
        for i in range(10):
            brep = get_brep(pred_model)
            text = format_prompt(prompt, SYSTEM_MESSAGE)
            with open(os.path.join(OUTPUT_DIR, "input_prompt.txt"), "w") as f:
                f.write(text[0])
            inputs = processor(text=text, breps=dgl.batch([brep]), max_length=3072).to(
                device
            )

            generated_value, generated_pointer, generated_scale = model.predict(
                mode="argmax" if not use_sampling else "sample", **inputs
            )
            pred_value, pred_pointer = generated_value[0], generated_pointer[0]

            pred_vector = list(zip(pred_value.tolist(), pred_pointer.tolist()))
            if pred_model.from_vector(pred_vector, 8):
                logger.info(
                    f"Generation task completed. The model contains {i+1} part(s) in total."
                )
                pred_model.normalize()
                mesh = create_mesh(pred_model)
                break
            logger.info(
                f"Generation of part {i+1} completed, continuing to generate the next part"
            )
        else:
            logger.warning(
                "Exceeded the maximum allowed number of parts. All generation attempts failed."
            )

    return mesh, pred_model


def convert_negative_zero(obj):
    if isinstance(obj, dict):
        return {k: convert_negative_zero(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_negative_zero(item) for item in obj]
    elif isinstance(obj, float) and obj == 0.0:
        return 0.0
    else:
        return obj


def remove_caption_and_visualization(obj):
    """
    Recursively remove any 'caption' or 'visualization' keys
    from nested dicts/lists.
    """
    if isinstance(obj, dict):
        keys_to_remove = [k for k in obj if k in ("caption", "visualization")]
        for k in keys_to_remove:
            del obj[k]
        return {k: remove_caption_and_visualization(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [remove_caption_and_visualization(item) for item in obj]

    else:
        return obj


def compact_numeric_arrays(json_str: str) -> str:
    # Match one-dimensional numeric arrays, allowing whitespace/newlines.
    pattern = r"\[\s*(?:-?\d+(?:\.\d+)?\s*,\s*)*-?\d+(?:\.\d+)?\s*\]"

    def compact_array(match):
        content = match.group()
        # Keep numeric values in original order.
        numbers = re.findall(r"-?\d+(?:\.\d+)?", content)
        return "[" + ", ".join(numbers) + "]"

    return re.sub(pattern, compact_array, json_str)


def generate_cad_model_from_text(text, use_sampling=False):
    if text is None or text.strip() == "":
        raise ValueError(
            "Text input cannot be empty. Please provide a valid text prompt."
        )

    mesh, cadmodel = test_model(
        model=model, prompt=text, device=device, use_sampling=use_sampling
    )
    if mesh is not None:
        output_path = os.path.join(OUTPUT_DIR, "output.stl")
        mesh.export(output_path)

        json_dict = cadmodel._json()
        json_dict = convert_negative_zero(json_dict)
        json_dict = remove_caption_and_visualization(json_dict)
        json_str = json.dumps(json_dict, indent=2, ensure_ascii=False)
        json_str = compact_numeric_arrays(json_str)

        return output_path, json_str
    else:
        raise Exception("Error generating CAD model from text")


def parse_config_file(config_file):
    with open(config_file, "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data


if not torch.cuda.is_available():
    logger.error("CUDA is not available. Please check your PyTorch installation.")

SYSTEM_MESSAGE = "You are an expert mechanical engineer. Based on the user's text requirements, generate the corresponding CAD model design."
config_path = "./config/web.yaml"
config = parse_config_file(config_path)
device = torch.device("cuda")
model = load_model(config, device)
processor: PointerCADProcessor = PointerCADProcessor.from_pretrained(
    pretrained_model_name_or_path=config["model"]["base_model"], padding_side="left"
)
time_str = datetime.datetime.now().strftime("%H:%M")
date_str = datetime.date.today()
OUTPUT_DIR = os.path.join(config["web"]["log_dir"], f"{date_str}/{time_str}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

examples = [
    [
        "Open a new sketch on the XY-plane in centimeters, draw a square with sides of 1 cm, then extrude it upward by 0.5 cm.",
        False,
    ],
    [
        "Start at (0 cm, 0 cm), draw to (8 cm, 0 cm), then to (6 cm, 3 cm), then to (2 cm, 3 cm), and back to (0 cm, 0 cm) counterclockwise. Extrude upward 2 cm.",
        False,
    ],
    [
        """Starting at (–5 cm, –5 cm), draw to (5 cm, –5 cm), then to (5 cm, 5 cm), then to (–5 cm, 5 cm), and back to (–5 cm, –5 cm) counterclockwise to form a 10 cm square. Add a circle with a radius of 2 cm at the center (0 cm, 0 cm), then extrude the shape 2 cm upward.""",
        False,
    ],
    [
        """Part 1: On the XY-plane, draw a circle with a radius of 10 cm and extrude it 10 cm along the positive normal direction.
Part 2: On the XY-plane, draw a circle with a radius of 5 cm and extrude it along the positive normal direction as a cut with a depth of 5 cm.""",
        False,
    ],
    [
        """Part 1: On the XY-plane, draw a circle with a radius of 10 cm and extrude it 10 cm along the positive normal direction.
Part 2: On the XY-plane rotated 180° around the X-axis, draw a circle with a radius of 5 cm and extrude it 5 cm along its positive normal direction.""",
        False,
    ],
    [
        """Part 1: On the XY-plane, draw a circle with a radius of 10 cm and extrude it 10 cm along the positive normal direction.
Part 2: On a sketch plane parallel to the XY-plane and translated 10 cm upward along the Z-axis, draw a circle with a radius of 5 cm and extrude it 5 cm.""",
        False,
    ],
    [
        """Draw two concentric circles:
In your modeling workspace, create two circles sharing the same center point. Set the outer circle’s radius to 10 cm and the inner circle’s radius to 8 cm. This will define the cross-section of the ring.

Select the area between the circles:
Highlight the ring-shaped region formed between the outer and inner circles. This will be the base profile for extrusion.

Extrude the profile:
Use the extrusion tool to extend the selected profile upward by 2 cm.

Complete the shape:
Confirm the operation to finalize the 3D ring with an outer radius of 10 cm, an inner radius of 8 cm, and a height of 2 cm.""",
        False,
    ],
]


title = "Pointer-CAD: Unifying B-Rep and Command Sequences via Pointer-based Edges & Faces Selection"
description = """
<div style="display: flex; justify-content: center; gap: 10px; align-items: center; margin-bottom: 12px;">

<a href="https://arxiv.org/abs/2603.04337">
    <img src="https://img.shields.io/badge/arXiv-2603.04337-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="Pointer-CAD arXiv" />
</a>
<a href="https://snitro.github.io/Pointer-CAD-Page/">
    <img src="https://img.shields.io/badge/Project-Page-7dd3fc?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Pointer-CAD Project Page" />
</a>
<img src="https://img.shields.io/badge/CVPR-2026%20Accepted-16a34a?style=for-the-badge&logo=googlescholar&logoColor=white&logoWidth=18" alt="Accepted by CVPR 2026" />

</div>

<p>
Recent LLM-based CAD generation methods represent models as command sequences, but they struggle to reference geometric entities (e.g., faces or edges), limiting complex edits such as chamfer and fillet and introducing quantization-induced topological errors.
</p>

<p>
We propose Pointer-CAD, a novel LLM-based framework that introduces a pointer-based command representation to explicitly model geometric relationships in B-rep CAD models. By conditioning each operation on both the textual description and the intermediate geometry, Pointer-CAD enables precise entity selection through pointer prediction, improving editing accuracy while reducing quantization errors.
</p>
"""

base_tmp = tempfile.gettempdir()
cache_path = os.path.join(base_tmp, f"gradio_{getpass.getuser()}")
os.makedirs(cache_path, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = cache_path

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Row():
        # Left column
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Text Prompt", placeholder="Enter a text prompt here"
            )
            use_sampling = gr.Checkbox(label="Use Sampling", value=False)
            # Generate button
            run_btn = gr.Button("Generate Model")

        # Right column
        with gr.Column(scale=2):
            output_3d = gr.Model3D(
                clear_color=[0.678, 0.847, 0.902, 1.0], label="3D CAD Model"
            )
            with gr.Accordion("Model JSON Data", open=False):
                output_json = gr.Textbox(
                    label="JSON", lines=20, interactive=False, show_copy_button=True
                )

    # Examples section (full width)
    gr.Examples(examples, [text_input, use_sampling])

    run_btn.click(
        fn=generate_cad_model_from_text,
        inputs=[text_input, use_sampling],
        outputs=[output_3d, output_json],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--port", type=int, default=7860, help="Port to launch the Gradio app"
    )
    args, unknown = parser.parse_known_args()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=True)

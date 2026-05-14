# Pointer-CAD source code


[![Paper](https://img.shields.io/badge/arXiv-2603.04337-b31b1b.svg)](https://arxiv.org/abs/2603.04337)
[![Project Page](https://img.shields.io/badge/WebPage-Pointer--CAD-blue)](https://snitro.github.io/Pointer-CAD-Page/)
[![Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/Snitro/Recap-OmniCAD)

This repository provides source code for our paper:

[**Pointer-CAD: Unifying B-Rep and Command Sequences via Pointer-based Edge & Face Selection**](https://arxiv.org/abs/2603.04337)

CVPR 2026

Pointer-CAD autoregressively generates parametric CAD models from natural language descriptions, constructing the model step by step. At each modeling step, it takes the full text prompt and the current intermediate B-Rep solid as input, then predicts the next CAD operation, such as extrude, fillet, or chamfer, while directly pointing to the faces or edges on the existing geometry referenced by that operation.

---

## How it works

Parametric CAD design is a sequence of operations. Each step takes the solid built so far and adds one more feature:

```
Natural language prompt
        │
        ▼
Current intermediate B-Rep solid
(initially empty)
        │
        ▼
Pointer-CAD
        │
        ▼
Next CAD operation
(extrude / fillet / chamfer)
        │
        ▼
Updated intermediate B-Rep solid
        │
        └── repeat step by step until the CAD model is complete
```

**Architecture overview:**

- **Language backbone**: Qwen2.5-Instruct, fine-tuned with LoRA (rank 8)
- **B-Rep encoder**: UV-Net — 1D/2D convolutions on UV-grid samples of edges and faces, followed by a graph neural network over the face-adjacency graph
- **Output heads**:
  - *Value head* — predict the next command token or 8-bit quantized numeric value
  - *Pointer head* — 128-dim embedding matched to B-Rep faces/edges via cosine similarity
  - *Scale head* — predict the overall model scale factor

---

## Prerequisites

- Linux
- NVIDIA GPU + CUDA CuDNN
- Python 3.10+, PyTorch 2.12+
- Conda (recommended)


## Repository structure

```
Pointer-CAD-dev/
├── config/          # Training, evaluation, and web demo hyperparameters
├── models/          # PointerCAD model, B-Rep encoder, and tokenizer
├── cadmodel/        # CAD feature definitions and OCC solid construction
├── dataset/         # Dataset loading and text prompt augmentation
├── preprocessing/   # JSON → token vector and B-Rep graph preprocessing
├── metrics/         # RL-oriented reward and metric implementations
└── measurements/    # Evaluation metrics (IoU, Chamfer, F1, watertightness)
```

---

## Installation

Install python package dependencies through pip:

```bash
pip install -r requirements.txt
```

Install the attention kernel (required for default `flash_attention_2` mode; compiles CUDA kernels):

```bash
pip install flash-attn --no-build-isolation
```

Install CAD geometry packages via conda:

```bash
conda install -c conda-forge pythonocc-core occwl
```

> **Note**: `occwl 3.0.0` is incompatible with `pythonocc-core 7.9.x` out of the box. Apply this patch after install:
> ```bash
> python -c "import site,pathlib; f=pathlib.Path(site.getsitepackages()[0])/'occwl/compound.py'; f.write_text(f.read_text().replace('import read_step_file, list_of_shapes_to_compound','import read_step_file\nfrom OCC.Core.BRep import BRep_Builder\ndef list_of_shapes_to_compound(s):\n from OCC.Core.TopoDS import TopoDS_Compound;c=TopoDS_Compound();b=BRep_Builder();b.MakeCompound(c);[b.Add(c,x) for x in s];return c,True'))"
> ```

Download the base model weights from Hugging Face (set in `config/train.yaml`):

```bash
# Default (fast, for debugging)
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct

# Full model (recommended for real training)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
```

---

## Data preparation

Download the dataset from Hugging Face and place it under `./data/pointercad`:

```bash
huggingface-cli download Snitro/Recap-OmniCAD \
    --repo-type dataset \
    --local-dir ./data/pointercad
````

After download, the data directory should contain the preprocessed Pointer-CAD dataset and the corresponding split file used for training and evaluation:

```text
data/pointercad/
├── dataset/
│   ├── 0000/
│   ├── 0001/
│   └── ...
└── train_val_test.json
````

---

## Training

Set paths and hyperparameters in `config/train.yaml`, then launch:

```bash
bash train.sh
```

**Distributed training environment:**

`train.sh` automatically detects the number of visible NVIDIA GPUs and uses it as the default `WORLD_SIZE`. If no GPU is detected, `WORLD_SIZE` defaults to `1`.

The following environment variables can be overridden before launch:

* `MASTER_ADDR` — master node address, default: `localhost`
* `MASTER_PORT` — master port, default: `32501`
* `WORLD_SIZE` — total number of distributed processes, default: detected GPU count
* `NODE_RANK` — rank of the current node, default: `0`
* `GLOBAL_RANK_OFFSET` — global rank offset, default: `0`

Example multi-node launch:

```bash
MASTER_ADDR=<master_ip> \
MASTER_PORT=32501 \
WORLD_SIZE=<total_processes> \
NODE_RANK=<node_rank> \
GLOBAL_RANK_OFFSET=<rank_offset> \
bash train.sh
```

Checkpoints are saved to `<log_dir>/<date>/<time>/model.pth`. Training is logged to WandB project `Pointer-CAD`.

If you do not want to use WandB, disable it in `config/train.yaml` by turning off the WandB-related option, or run training with WandB disabled from the command line:

```bash
WANDB_MODE=disabled bash train.sh
```

You can also use offline logging if you want to keep local WandB logs without syncing them to the WandB server:

```bash
WANDB_MODE=offline bash train.sh
```

In offline mode, logs are saved locally and can be synced later with `wandb sync`.

---

## Web demo

Set `config/web.yaml`:

```yaml
model:
  base_model: "Qwen/Qwen2.5-7B-Instruct"
web:
  checkpoint_path: "./log/2025-01-01/12:00/model.pth"
  log_dir: "./log"
```

Launch:

```bash
python web.py
```

The web interface accepts a text description and generates the corresponding CAD model step by step.

---

## Exporting CAD files

Convert raw JSON models to other formats without running the full pipeline:

```bash
# Export to STEP
python preprocessing/json2step.py --input_dir ./data/raw_json --output_dir ./exports/step

# Export to STL mesh
python preprocessing/json2stl.py --input_dir ./data/raw_json --output_dir ./exports/stl
```

---

## Citation

```bibtex
@inproceedings{pointercad2026,
  title     = {Pointer-CAD: Unifying B-Rep and Command Sequences via Pointer-based Edges \& Faces Selection},
  author    = {Dacheng Qi and Chenyu Wang and Jingwei Xu and Tianzhe Chu and Zibo Zhao and Wen Liu and Wenrui Ding and Yi Ma and Shenghua Gao},
  booktitle = {CVPR},
  year      = {2026}
}
```

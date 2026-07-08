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
- Python 3.9, PyTorch 2.4
- Conda

---


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

## Environment Setup

This project was developed and tested under the following environment. We recommend using **Conda** to manage dependencies.

### 1. Create Conda Environment

Create a new Conda environment named `PointerCAD`:

```bash
conda create -n PointerCAD python=3.9
conda activate PointerCAD
```

### 2. Install Core Dependencies

Install CAD-related libraries and Python dependencies:

```bash
conda install pythonocc-core=7.7.0 loguru trimesh simpleeval scikit-learn -c conda-forge
```

### 3. Install PyTorch with CUDA Support

The project uses PyTorch 2.4.1 with CUDA 12.4 support:

```bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

### 4. Install Transformer and Graph Learning Dependencies

Install HuggingFace Transformers, PEFT, and DGL:

```bash
conda install transformers=4.51.3 peft=0.15.2 -c conda-forge

conda install -c dglteam/label/th24_cu124 dgl
```

### 5. Install pip Dependencies

Install `occwl` and additional Python packages:

```bash
pip install git+https://github.com/AutodeskAILab/occwl.git

pip install deprecate==1.0.5
pip install opencv-python
pip install wandb
```

### 6. Install FlashAttention

This project requires FlashAttention 2.7.4.

Instead of compiling from source, you can directly install the pre-built wheel:

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
```

The wheel above is built for:

* Python: 3.9
* PyTorch: 2.4.x
* CUDA: 12.x
* Linux x86_64

If your CUDA/PyTorch/Python versions are different, please install FlashAttention from source following the official instructions:
[https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

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
```

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
@inproceedings{qi2026pointer,
  title={Pointer-cad: Unifying b-rep and command sequences via pointer-based edges \& faces selection},
  author={Qi, Dacheng and Wang, Chenyu and Xu, Jingwei and Chu, Tianzhe and Zhao, Zibo and Liu, Wen and Ding, Wenrui and Ma, Yi and Gao, Shenghua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17377--17387},
  year={2026}
}
```

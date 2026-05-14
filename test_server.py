import os
import json
import yaml
import time
import uvicorn
import threading
from tqdm import tqdm
from loguru import logger
from statistics import median
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

app = FastAPI()


# Request body schemas
class ConnectRequest(BaseModel):
    gpu_info: str
    dataset_size: int
    log_dir: str
    config: dict


class ModelIDRequest(BaseModel):
    model_id: str


class ReportRequest(BaseModel):
    model_id: str
    result: dict


class FinishRequest(BaseModel):
    gpu_info: str


# Global runtime state
clients = []
claimed_model_ids = {}
invalid_model_ids = []
mious = []
cds = []
f1 = {"line": [], "circle": [], "arc": [], "extrude": [], "chamfer": [], "fillet": []}
watertightness = []
global_dataset_size = None
progress_bar = None
log_dir = None
finished_clients = 0

logger.remove()
logger.add(
    lambda msg: tqdm.write(msg, end=""),
    level="INFO",
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
)


@app.post("/connect")
async def connect(req: ConnectRequest):
    global global_dataset_size, progress_bar, log_dir

    if global_dataset_size is None:
        global_dataset_size = req.dataset_size
        progress_bar = tqdm(
            total=global_dataset_size, dynamic_ncols=True, desc="Test ✨"
        )
        logger.info(f"Initial dataset_size set to {global_dataset_size}")
    elif req.dataset_size != global_dataset_size:
        raise HTTPException(status_code=400, detail="Dataset size mismatch with server")

    if log_dir is None:
        log_dir = req.log_dir
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"job_{len(clients)}.yaml"), "w+") as f:
        yaml.dump(req.config, f, default_flow_style=False)

    clients.append(req.gpu_info)
    logger.info(f"Client connected: GPU={req.gpu_info}, Total clients={len(clients)}")
    return {"client_count": len(clients), "status": "connected", "log_dir": log_dir}


@app.post("/apply_model_id")
async def apply_model_id(req: ModelIDRequest):
    already_claimed = req.model_id in claimed_model_ids
    allowed = not already_claimed
    if allowed:
        claimed_model_ids[req.model_id] = None
        logger.info(
            f"Model ID '{req.model_id}' approved. Total claimed: {len(claimed_model_ids)}"
        )

    return {
        "allowed": allowed,
        "model_id": req.model_id,
        "claimed": len(claimed_model_ids),
        "total": global_dataset_size,
        "progress": round(len(claimed_model_ids) / global_dataset_size, 4),
    }


@app.post("/report")
async def report_result(req: ReportRequest):
    global progress_bar, claimed_model_ids, invalid_model_ids, mious, cds, f1, watertightness

    if req.model_id not in claimed_model_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Model ID '{req.model_id}' not found or not approved.",
        )

    if progress_bar:
        progress_bar.update(1)

    claimed_model_ids[req.model_id] = req.result

    if not req.result.get("status", True):
        invalid_model_ids.append(req.model_id)
    else:
        if (
            "intersection over union" in req.result
            and req.result["intersection over union"] is not None
        ):
            mious.append(req.result["intersection over union"])
        if (
            "chamfer distance" in req.result
            and req.result["chamfer distance"] is not None
        ):
            cds.append(req.result["chamfer distance"])
        if "f1" in req.result:
            for key in f1.keys():
                if key in req.result["f1"] and req.result["f1"][key] is not None:
                    f1[key].append(req.result["f1"][key])
        if "is watertight" in req.result and req.result["is watertight"] is not None:
            watertightness.append(int(req.result["is watertight"]))

    progress_bar.set_postfix(
        {
            "mIoU": f"{(sum(mious) / len(mious)):.2f}%" if mious else "N/A",
            "CDmean": f"{(sum(cds) / len(cds)):.2f}" if cds else "N/A",
            "CDmedian": f"{median(cds):.2f}" if cds else "N/A",
            "LineF1": (
                f"{(sum(f1['line']) / len(f1['line'])):.2f}%" if f1["line"] else "N/A"
            ),
            "ArcF1": (
                f"{(sum(f1['arc']) / len(f1['arc'])):.2f}%" if f1["arc"] else "N/A"
            ),
            "CircleF1": (
                f"{(sum(f1['circle']) / len(f1['circle'])):.2f}%"
                if f1["circle"]
                else "N/A"
            ),
            "ExtrudeF1": (
                f"{(sum(f1['extrude']) / len(f1['extrude'])):.2f}%"
                if f1["extrude"]
                else "N/A"
            ),
            "ChamferF1": (
                f"{(sum(f1['chamfer']) / len(f1['chamfer'])):.2f}%"
                if f1["chamfer"]
                else "N/A"
            ),
            "FilletF1": (
                f"{(sum(f1['fillet']) / len(f1['fillet'])):.2f}%"
                if f1["fillet"]
                else "N/A"
            ),
            "WT": (
                f"{(sum(watertightness) / len(watertightness)) * 100:.2f}%"
                if watertightness
                else "N/A"
            ),
            "IR": (
                f"{len(invalid_model_ids) / progress_bar.n * 100:.2f}%"
                if progress_bar.n > 0
                else "N/A"
            ),
        }
    )

    logger.info(f"Received report for Model ID '{req.model_id}': {req.result}")
    return {"status": "report received", "model_id": req.model_id}


@app.post("/finish")
async def finish(req: FinishRequest):
    global finished_clients, log_dir, claimed_model_ids

    finished_clients += 1
    logger.info(
        f"Client finished: {req.gpu_info}. Total finished: {finished_clients}/{len(clients)}"
    )

    output_path = os.path.join(log_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(claimed_model_ids, f, indent=4)
        logger.success(f"Results saved to {output_path}")

    if finished_clients >= len(clients):
        logger.info("All clients finished. Shutting down server...")

        def shutdown():
            time.sleep(0.5)
            os._exit(0)

        threading.Thread(target=shutdown, daemon=True).start()

    return {
        "status": "acknowledged",
        "finished": finished_clients,
        "total_clients": len(clients),
    }


@app.get("/ping")
async def ping():
    return {"status": "pong", "clients": len(clients), "finished": finished_clients}


if __name__ == "__main__":
    logger.info("Starting FastAPI server with loguru and tqdm support...")
    uvicorn.run(app, host="0.0.0.0", port=32500, log_config=None, access_log=False)

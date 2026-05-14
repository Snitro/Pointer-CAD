import pandas as pd
import statistics
import os, sys
import argparse

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))
from tqdm import tqdm
import traceback
from rich import print
from loguru import logger
import json
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("-i", "--input_path", default="./log", help="Predicted result")

    args = parser.parse_args()

    logger.info("Evaluation for Design History")

    json_path = None
    if os.path.isdir(args.input_path):
        json_pathes = []
        for root, dirs, files in os.walk(args.input_path):
            for file in files:
                if file.endswith(".json"):
                    json_pathes.append(os.path.join(root, file))

        json_path = sorted(json_pathes)[-1]
    else:
        json_path = args.input_path

    logger.info(f"Loading results from {json_path}")
    with open(json_path, "r") as fp:
        results: dict = json.load(fp)

    error_message = []
    intersection_over_union = []
    chamfer_distance = []
    line_f1 = []
    arc_f1 = []
    circle_f1 = []
    extrude_f1 = []
    chamfer_f1 = []
    fillet_f1 = []
    watertightness = []

    for model_id, data in results.items():
        if data is not None and data["status"]:
            if data["intersection over union"] is not None:
                intersection_over_union.append(data["intersection over union"])
            if data["chamfer distance"] is not None:
                chamfer_distance.append(data["chamfer distance"])
            if data["f1"] is not None:
                if "line" in data["f1"]:
                    line_f1.append(data["f1"]["line"])
                if "arc" in data["f1"]:
                    arc_f1.append(data["f1"]["arc"])
                if "circle" in data["f1"]:
                    circle_f1.append(data["f1"]["circle"])
                if "extrude" in data["f1"]:
                    extrude_f1.append(data["f1"]["extrude"])
                if "chamfer" in data["f1"]:
                    chamfer_f1.append(data["f1"]["chamfer"])
                if "fillet" in data["f1"]:
                    fillet_f1.append(data["f1"]["fillet"])
            if data["is watertight"] is not None:
                watertightness.append(data["is watertight"])
        else:
            if data is None:
                error_message.append("No response")
            else:
                error_message.append(data["error_message"])

    eval_dict = {}

    eval_dict["failure"] = {}
    eval_dict["failure"]["rate"] = (
        len(error_message) / len(results) if len(results) > 0 else 0
    ) * 100
    error_types = error_message.copy()
    error_counter = Counter(error_types)
    eval_dict["failure"]["detail"] = {}
    total_failures = len(error_types)
    for error, count in error_counter.items():
        eval_dict["failure"]["detail"][error] = (
            count / total_failures if total_failures > 0 else 0
        ) * 100

    eval_dict["intersection over union"] = {}
    eval_dict["intersection over union"]["median"] = statistics.median(
        intersection_over_union
    )
    eval_dict["intersection over union"]["mean"] = statistics.mean(
        intersection_over_union
    )

    eval_dict["chamfer distance"] = {}
    eval_dict["chamfer distance"]["median"] = statistics.median(chamfer_distance)
    eval_dict["chamfer distance"]["mean"] = statistics.mean(chamfer_distance)

    eval_dict["f1"] = {
        "line": statistics.mean(line_f1),
        "arc": statistics.mean(arc_f1),
        "circle": statistics.mean(circle_f1),
        "extrude": statistics.mean(extrude_f1),
        "chamfer": statistics.mean(chamfer_f1) if len(chamfer_f1) > 0 else "N/A",
        "fillet": statistics.mean(fillet_f1) if len(fillet_f1) > 0 else "N/A",
    }

    eval_dict["watertightness"] = statistics.mean(watertightness) * 100

    json_formatted_str = json.dumps(eval_dict, indent=4)
    print("\n\n")
    print("=" * 10, "Evaluation Results", "=" * 10)
    print(json_formatted_str)
    print("=" * 40)
    print("\n\n")

    report_path = os.path.join(os.path.dirname(json_path), "report.json")
    with open(report_path, "w") as f:
        json.dump(eval_dict, f, indent=4)

    logger.success(
        f"Evaluation completed: evaluated {len(results)} items. Results saved to {report_path}"
    )


if __name__ == "__main__":
    main()

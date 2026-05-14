import os
import sys
import time
import json
import argparse

sys.path.append("..")

from tqdm import tqdm
from itertools import repeat
from multiprocessing import Process
from cadmodel.model import CADModel, convert_json_from_deepcad
from occwl.io import save_stl
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh


def process_one_file(json_path, output_dir):
    file_name = os.path.splitext(os.path.basename(json_path))[0]
    with open(json_path, "r") as fp:
        data = json.load(fp)
        if "properties" in data:
            data = convert_json_from_deepcad(data)

    model = CADModel.from_dict(data)
    model.normalize()

    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(model.seq)):
        model_sub = model.submodel(i)
        model_sub.normalize(keep_sketch_plane=False)

        shape = model_sub.build_model().topods_shape()
        mesh = BRepMesh_IncrementalMesh(shape, 0.001, False, 0.5, True)
        mesh.Perform()
        if not mesh.IsDone():
            raise RuntimeError("Mesh generation failed.")
        save_stl(
            shape, os.path.join(output_dir, f"{file_name}_{model.seq[i].index:05d}.stl")
        )


def multiprocessing(arguments):
    try:
        input_file, args = arguments
        process_one_file(input_file, os.path.join(args.output_dir, f"{args.chunk:04d}"))
    except Exception as e:
        tqdm.write(f"Error: {os.path.basename(input_file)} --- {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Process command line arguments")

    parser.add_argument(
        "-c", "--chunk", type=int, default=0, help="Chunk value, default is 0"
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Input directory path",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Output directory path",
    )
    parser.add_argument(
        "-n", "--num_processes", type=int, default=16, help="Number of processes to use"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Maximum runtime for each task (seconds)",
    )

    args = parser.parse_args()

    input_dir = os.path.join(args.input_dir, f"{args.chunk:04d}")
    files_and_folders = os.listdir(input_dir)
    input_files = [
        os.path.join(input_dir, f)
        for f in files_and_folders
        if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".json")
    ]

    max_workers = args.num_processes
    processes = []
    pending = list(zip(input_files, repeat(args)))

    try:
        with tqdm(total=len(pending)) as pbar:
            while pending or processes:
                while pending and len(processes) < max_workers:
                    task_args = pending.pop(0)
                    p = Process(target=multiprocessing, args=(task_args,))
                    p.start()
                    processes.append((p, time.time()))

                still_running = []
                for proc, start_time in processes:
                    proc.join(timeout=0)
                    if proc.is_alive():
                        if time.time() - start_time > args.timeout:
                            tqdm.write(
                                f"Task timeout, terminating process PID {proc.pid}"
                            )
                            proc.terminate()
                            proc.join()
                            pbar.update(1)
                        else:
                            still_running.append((proc, start_time))
                    else:
                        pbar.update(1)
                processes = still_running
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Detected Ctrl+C, terminating all processes...")
        for proc, _ in processes:
            proc.terminate()
        for proc, _ in processes:
            proc.join()


if __name__ == "__main__":
    main()

import os
import sys
import time
import json
import pickle
import argparse

sys.path.append("..")

from tqdm import tqdm
from itertools import repeat
from multiprocessing import Process
from cadmodel.model import CADModel, convert_json_from_deepcad
from dgl.data.utils import save_graphs


def process_one_file(
    json_path, output_dir, curv_u_samples, surf_u_samples, surf_v_samples, quant_bits
):
    with open(json_path, "r") as fp:
        data = json.load(fp)
        if "properties" in data:
            data = convert_json_from_deepcad(data)

    base_name: str = os.path.splitext(os.path.basename(json_path))[0]
    if "_" in base_name:
        base_name = base_name.split("_")[0]

    model = CADModel.from_dict(data)
    model.normalize()

    for i in range(len(model.seq)):
        graph, vec, scale, movement = model.to_vector(
            i, curv_u_samples, surf_u_samples, surf_v_samples, quant_bits
        )
        output_path_vector = os.path.join(
            output_dir, base_name, "vector", f"{base_name}_{model.seq[i].index:05d}.pkl"
        )
        output_path_graph = os.path.join(
            output_dir, base_name, "graph", f"{base_name}_{model.seq[i].index:05d}.bin"
        )

        os.makedirs(os.path.dirname(output_path_vector), exist_ok=True)
        os.makedirs(os.path.dirname(output_path_graph), exist_ok=True)

        vector = {"vec": vec, "scale": scale, "movement": movement}

        with open(output_path_vector, "wb") as f:
            pickle.dump(vector, f)

        if graph is not None:
            save_graphs(output_path_graph, graph)


def multiprocessing(arguments):
    try:
        input_file, args = arguments
        process_one_file(
            input_file,
            os.path.join(args.output_dir, f"{args.chunk:04d}"),
            args.curv_u_samples,
            args.surf_u_samples,
            args.surf_v_samples,
            args.quant_bits,
        )
    except Exception as e:
        tqdm.write(f"Error: {os.path.basename(input_file)} --- {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Process command-line arguments")

    parser.add_argument("-c", "--chunk", type=int, default=0, help="Chunk index (default: 0)")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--curv_u_samples", type=int, default=32, help="curv_u_samples value"
    )
    parser.add_argument(
        "--surf_u_samples", type=int, default=32, help="surf_u_samples value"
    )
    parser.add_argument(
        "--surf_v_samples", type=int, default=32, help="surf_v_samples value"
    )
    parser.add_argument("--quant_bits", type=int, default=8, help="quant_bits value")
    parser.add_argument(
        "--num_processes", type=int, default=16, help="Number of processes to use"
    )
    parser.add_argument(
        "--timeout", type=int, default=600, help="Maximum runtime per task (seconds)"
    )

    args = parser.parse_args()

    input_dir = os.path.join(args.input_dir, f"{args.chunk:04d}")
    files_and_folders = os.listdir(input_dir)
    input_files = sorted(
        [
            f
            for f in files_and_folders
            if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".json")
        ],
        reverse=True,
    )
    is_deepcad = "_" in input_files[0]

    if is_deepcad:
        input_files_clean = []
        for i in range(len(input_files)):
            base_name = input_files[i].split("_")[0]
            if i == 0 or base_name != input_files[i - 1].split("_")[0]:
                input_files_clean.append(input_files[i])
        input_files = input_files_clean
    input_files = [os.path.join(input_dir, f) for f in input_files]

    max_workers = args.num_processes
    processes: list[tuple[Process, float]] = []
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
                            tqdm.write(f"Task timeout, terminating process PID {proc.pid}")
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

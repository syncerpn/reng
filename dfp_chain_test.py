import argparse
import os
import json
import subprocess
import re

FPS_PATTERN = "  Average FPS: "

def parse_test_output(filename):
    data = []
    with open(filename, "r") as f:
        data = f.readlines()

    data = [d.strip('\n') for d in data]
    fps = -1
    for d in data:
        i = d.find(FPS_PATTERN)
        if i >= 0:
            fps = float(re.findall(r'\d+\.\d+', d)[0])
            break
        
    return fps

def run_test_command(command, run_output_file, skip_existed):
    command_executed = False

    if not skip_existed or not os.path.exists(run_output_file):
        try:
            print(f"[info] running command: {command}")
            with open(run_output_file, "w") as f:
                subprocess.call(command, shell=True, stdout=f, timeout=100)
            command_executed = True
        except subprocess.CalledProcessError as e:
            print(f"[warn] command failed: {e}")

    # post-test verification
    fps = -1
    if not os.path.exists(run_output_file):
        print(f"[warn] run log not found: {run_output_file}")
    else:
        fps = parse_test_output(run_output_file)

    return command_executed, fps

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-path", type=str, required=True, help="onnx meta data file path")
    parser.add_argument("--mx_bench", type=str, default="mx_bench", help="memryx benchmark binary path")
    parser.add_argument("--save-dir", type=str, help="path to save the test logs")
    parser.add_argument("--skip-existed", action="store_true", help="continue previous test")

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    save_dir = args.save_dir if args.save_dir else os.path.join(args.meta_path, "dfp_test")

    assert args.skip_existed or not os.path.exists(save_dir), "[info] dfp_test saving path already exists"
    os.makedirs(save_dir, exist_ok=True)

    meta_data = {}
    test_output_file = "test.json"

    with open(os.path.join(args.meta_path, "meta.json"), "r") as json_file:
        meta_data = json.load(json_file)

    for i, model_name in enumerate(meta_data):
        benchmark_commands = {
            "f_1000": f"{args.mx_bench} -v -f 1000 -d",
        }

        test_names = ["f_1000"]
        fpss = [-1] * len(test_names)

        print(f"[info][{i+1}/{len(meta_data)}] model: {model_name}")

        for ti, test_name in enumerate(test_names):
            run_output_file = os.path.join(save_dir, f"{model_name}.{test_name}.log")
            dfp_file = os.path.join(args.meta_path, "dfp", model_name + ".dfp")

            # perform single test first to decide how many loops for multi-core tests
            command = f"{benchmark_commands[test_name]} {dfp_file}"

            command_executed, fps = run_test_command(command, run_output_file, args.skip_existed)

            fpss[ti] = fps

            if fps == -1:
                print(f"[warn] error detected")
                assert command_executed, "[info] test stops due to a command execution error; please check the error"

        meta_data[model_name]["test"] = {"name": test_names, "fps": fpss}

        with open(os.path.join(save_dir, test_output_file), "w") as json_file:
            json.dump(meta_data, json_file)

if __name__ == "__main__":
    main()

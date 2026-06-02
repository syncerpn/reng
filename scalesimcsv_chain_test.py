import argparse
import os
import json
import subprocess
import dxnn_utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-path", type=str, required=True, help="onnx meta data file path")
    parser.add_argument("--dxrt", type=str, required=True, help="dxrt binary path")
    parser.add_argument("--break-time", type=int, default=10, help="break between runs to ensure device stability")
    parser.add_argument("--save-dir", type=str, help="path to save the test logs")
    parser.add_argument("--skip-existed", action="store_true", help="continue previous test")
    parser.add_argument("--soft-restart", action="store_true", help="soft restart the device")

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    save_dir = args.save_dir if args.save_dir else os.path.join(args.meta_path, "dxnn_test")

    assert args.skip_existed or not os.path.exists(save_dir), "[info] dxnn_test saving path already exists"
    os.makedirs(save_dir, exist_ok=True)

    meta_data = {}

    with open(os.path.join(args.meta_path, "meta.json"), "r") as json_file:
        meta_data = json.load(json_file)

    for i, model_name in enumerate(meta_data):
        run_output_file = os.path.join(save_dir, model_name + ".log")
        fps, mem = 0.0, 0

        command_executed = False

        if not args.skip_existed or not os.path.exists(run_output_file):
            dxnn_file = os.path.join(args.meta_path, "dxnn", model_name + ".dxnn")

            command = f"{args.dxrt} -l 100 -m {dxnn_file} \
                && rm profiler.json && sleep {args.break_time}"

            try:
                print(f"[info][{i+1}/{len(meta_data)}] running test: {command}")
                with open(run_output_file, "w") as f:
                    subprocess.call(command, shell=True, stdout=f, timeout=100)
                command_executed = True
            except subprocess.CalledProcessError as e:
                print(f"[warn] command failed: {e}")

        # post-test verification
        if not os.path.exists(run_output_file):
            print(f"[warn] run log not found: {model_name}")

        fps, mem = dxnn_utils.parse_test_output(run_output_file)
        if fps == -1:
            print(f"[warn] error detected")
            if command_executed:
                if args.soft_restart:
                    print(f"[info] try soft restart the device and continue the test")
                    subprocess.call(f"dxrt-cli -r 0 && sudo service dxrt restart && sleep 3", shell=True)
                else:
                    assert not command_executed, "[info] test stops due to a command execution error; please check the error"

        meta_data[model_name]["test"] = {"fps": fps, "mem": mem}

        with open(os.path.join(save_dir, "test.json"), "w") as json_file:
            json.dump(meta_data, json_file)

if __name__ == "__main__":
    main()

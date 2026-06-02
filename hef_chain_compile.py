import argparse
import os
import json
import subprocess

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-path", type=str, required=True, help="onnx meta data file path")
    parser.add_argument("--hailo", type=str, required=True, help="hailo binary path")
    parser.add_argument("--hw-arch", type=str, default="hailo8", help="hailo binary path")
    parser.add_argument("--save-dir", type=str, help="path to save the generated models")
    parser.add_argument("--skip-existed", action="store_true", help="continue previous compilation")

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    save_dir = args.save_dir if args.save_dir else os.path.join(args.meta_path, "hef")

    assert args.skip_existed or not os.path.exists(save_dir), "[info] hef saving path already exists"
    os.makedirs(save_dir, exist_ok=True)

    meta_data = {}

    with open(os.path.join(args.meta_path, "meta.json"), "r") as json_file:
        meta_data = json.load(json_file)

    for i, model_name in enumerate(meta_data):
        hef_file = os.path.join(save_dir, model_name + ".hef")
        if args.skip_existed and os.path.exists(hef_file):
            print(f"[info][{i+1}/{len(meta_data)}] skipped, due to already compiled: {model_name}")
            continue

        onnx_file = os.path.join(args.meta_path, "onnx", model_name + ".onnx")

        command = f"{args.hailo} parser onnx {onnx_file} --hw-arch {args.hw_arch} \
            && {args.hailo} optimize {model_name}.har --use-random-calib-set \
            && {args.hailo} compiler {model_name}_optimized.har --hw-arch {args.hw_arch} \
            && mv {model_name}.hef {save_dir} \
            && rm {model_name}.har \
            && rm {model_name}_optimized.har \
            && rm {model_name}_compiled.har \
            && rm allocator.log \
            && rm acceleras.log \
            && rm hailo_sdk*.log "

        try:
            print(f"[info][{i+1}/{len(meta_data)}] running compilation: {command}")
            subprocess.call(command, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"[warn] command failed: {e}")

        # post-compilation verification
        if not os.path.exists(hef_file):
            print(f"[warn] model compilation failed {model_name}")

if __name__ == "__main__":
    main()

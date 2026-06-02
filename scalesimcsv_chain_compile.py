import argparse
import os
import json
import subprocess
import architecture

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-path", type=str, required=True, help="onnx meta data file path")
    parser.add_argument("--save-dir", type=str, help="path to save the generated models")
    parser.add_argument("--skip-existed", action="store_true", help="continue previous compilation")

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    save_dir = args.save_dir if args.save_dir else os.path.join(args.meta_path, "scalesimcsv")

    assert args.skip_existed or not os.path.exists(save_dir), "[info] scalesimcsv saving path already exists"
    os.makedirs(save_dir, exist_ok=True)

    meta_data = {}

    with open(os.path.join(args.meta_path, "meta.json"), "r") as json_file:
        meta_data = json.load(json_file)

    for i, model_name in enumerate(meta_data):

        csv_file = os.path.join(save_dir, model_name + ".csv")
        if args.skip_existed and os.path.exists(csv_file):
            print(f"[info][{i+1}/{len(meta_data)}] skipped, due to already compiled: {model_name}")
            continue

        model_csv_stats = architecture.parse(model_name)
        print(f"[info][{i+1}/{len(meta_data)}] compiling model csv stats: {model_name}")
        with open(csv_file, "w") as file:
            file.write("".join([",".join(list(map(str, d))) + "\n" for d in model_csv_stats]))

        # post-compilation verification
        if not os.path.exists(csv_file):
            print(f"[warn] model compilation failed {model_name}")

if __name__ == "__main__":
    main()

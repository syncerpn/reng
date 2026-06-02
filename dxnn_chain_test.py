import argparse
import os
import json
import subprocess

FPS_PATTERN = "- FPS"
FPS_PATTERN_SEP = ":"
MEM_PATTERN = "bytes "

def parse_test_output(filename):
    data = []
    with open(filename, "r") as f:
        data = f.readlines()

    data = [d.strip('\n') for d in data]
    fps, mem = -1, 0
    for d in data:
        i = d.find(FPS_PATTERN)
        if i >= 0:
            j = d.find(FPS_PATTERN_SEP)
            fps = float(d[j + len(FPS_PATTERN_SEP):])
            break
        
        j = d.find(MEM_PATTERN)
        if j >= 0:
            mem = int(d[:j].split(" ")[-1])

    return fps, mem

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
    fps, mem = -1, 0
    if not os.path.exists(run_output_file):
        print(f"[warn] run log not found: {run_output_file}")
    else:
        fps, mem = parse_test_output(run_output_file)

    return command_executed, fps, mem

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
    test_output_file = "test.json"

    cleanup_command = f"rm profiler.json"
    break_command = f"sleep {args.break_time}"
    failsafe_command = f"dxrt-cli -r 0 && sudo service dxrt restart && sleep 3"

    with open(os.path.join(args.meta_path, "meta.json"), "r") as json_file:
        meta_data = json.load(json_file)

    for i, model_name in enumerate(meta_data):
        benchmark_commands = {
            "single"  : f"{args.dxrt} -s -m",
            "all"     : f"{args.dxrt} -b -n 0 --skip-io -m",
            "core_0"  : f"{args.dxrt} -b -n 1 --skip-io -m",
            "core_1"  : f"{args.dxrt} -b -n 2 --skip-io -m",
            "core_2"  : f"{args.dxrt} -b -n 3 --skip-io -m",
            "core_0_1": f"{args.dxrt} -b -n 4 --skip-io -m",
            "core_1_2": f"{args.dxrt} -b -n 5 --skip-io -m",
            "core_0_2": f"{args.dxrt} -b -n 6 --skip-io -m",
        }

        test_names = ["single","all","core_0","core_1","core_2","core_0_1","core_1_2","core_0_2",]
        fpss = [-1] * len(test_names) # 8 tests
        mems = [ 0] * len(test_names) # 8 tests

        loop_option = ""

        print(f"[info][{i+1}/{len(meta_data)}] model: {model_name}")

        for ti, test_name in enumerate(test_names):
            run_output_file = os.path.join(save_dir, model_name + f".{test_name}.log")
            dxnn_file = os.path.join(args.meta_path, "dxnn", model_name + ".dxnn")

            # perform single test first to decide how many loops for multi-core tests
            command = f"{benchmark_commands[test_name]} {dxnn_file} {loop_option} && {cleanup_command} && {break_command}"

            command_executed, fps, mem = run_test_command(command, run_output_file, args.skip_existed)

            fpss[ti] = fps
            mems[ti] = mem

            if fps == -1:
                print(f"[warn] error detected")
                if command_executed:
                    if args.soft_restart:
                        print(f"[info] try soft restart the device and continue the test")
                        subprocess.call(failsafe_command, shell=True)
                    else:
                        assert not command_executed, "[info] test stops due to a command execution error; please check the error"

                if not loop_option:
                    print(f"[info] single-core sequential test failed; stop all other tests")
                    break

            if not loop_option:
                loop_option = f"-l {min(10000, int(fps * 30))}"
                print(f"[info] run benchmark test with {loop_option}")

        meta_data[model_name]["test"] = {"name": test_names, "fps": fpss, "mem": mems}

        with open(os.path.join(save_dir, test_output_file), "w") as json_file:
            json.dump(meta_data, json_file)

if __name__ == "__main__":
    main()

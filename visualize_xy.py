import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=str, required=True, help="report file")
    parser.add_argument("--x", type=str, required=True, help="x data header")
    parser.add_argument("--y", type=str, required=True, help="y data header")
    parser.add_argument("--name", type=str, help="plot name")
    parser.add_argument("--save-dir", default="./", type=str, help="figure save dir")

    args = parser.parse_args()

    return args

HEADER = ["name", "n_layers", "params", "ops", "mems_iop", "mems_wfmap", "mem_test", "operational_intensity_iop", "operational_intensity_wfmap", "operational_intensity_test", "fps", "throughput"]

def main():
    args = get_args()

    raw_data = []

    with open(args.report, "r") as f:
        raw_data = [d.strip() for d in f.readlines()]

    header = raw_data[0].split(",")
    assert args.x in header, f"[erro] x header not found in the report"
    assert args.y in header, f"[erro] y header not found in the report"

    raw_data = list(zip(*[d.split(",") for d in raw_data[1:]]))

    data = {}

    for i, h in enumerate(header):
        try:
            data[h] = list(map(float, raw_data[i]))
        except:
            data[h] = raw_data[i]

    x_data = data[args.x]
    y_data = data[args.y]

    plt.figure(figsize=(12, 6))
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.rc('axes', axisbelow=True)

    plt.scatter(x_data, y_data, color="b", alpha=0.2)

    # Labels
    plt.xlabel(args.x)
    plt.ylabel(args.y)
    plt.title(f"{args.name} Data correlation")

    os.makedirs(args.save_dir, exist_ok=True)

    plt.savefig(os.path.join(args.save_dir, f"xy_{args.name}_{args.x}_{args.y}.svg"))
    print(f"[info] plot saved as xy_{args.name}_{args.x}_{args.y}.svg")

if __name__ == "__main__":
    main()

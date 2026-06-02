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
    parser.add_argument("--max-throughput", type=float, required=True, help="device max throughput (GOPS)")
    parser.add_argument("--memory-bandwidth", type=float, required=True, help="memory bandwidth (GB/s)")
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
    assert args.x in header, f"[erro] x header {args.x} not found in the report"
    assert args.y in header, f"[erro] y header {args.y} not found in the report"

    raw_data = list(zip(*[d.split(",") for d in raw_data[1:]]))

    data = {}

    for i, h in enumerate(header):
        try:
            data[h] = list(map(float, raw_data[i]))
        except:
            data[h] = raw_data[i]

    P_max = args.max_throughput
    B_max = args.memory_bandwidth

    x_data = data[args.x]
    y_data = data[args.y]

    max_test_bw = max([yi/xi for xi, yi in zip(x_data, y_data)])
    max_test_tp = max(y_data)
    
    x_tick_min = min(0, math.log10(min(x_data)))
    x_tick_max = max(6, math.log10(max(x_data)))
    x_ridge_log = math.log10(P_max / B_max)
    
    if x_tick_max + x_tick_min > 2 * x_ridge_log:
        x_tick_min = 2 * x_ridge_log - x_tick_max
    else:
        x_tick_max = 2 * x_ridge_log - x_tick_min

    OI = np.logspace(x_tick_min, x_tick_max, 100)

    plt.figure(figsize=(12, 6))
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.rc('axes', axisbelow=True)

    performance = np.minimum(P_max, OI * B_max)
        
    plt.loglog(OI, performance, label=f"roofline limit", linewidth=2, color="r")
    plt.axvline(x=P_max / B_max, linestyle="--", color="green", label=f"ridge point")

    plt.scatter(x_data, y_data, color="b", alpha=0.2)

    # Labels
    plt.xlabel("Operational Intensity (OPs/Byte)")
    plt.ylabel("Performance (GOPS)")
    plt.title(f"{args.name} Roofline Model - bandwidth: {max_test_bw:,.2f} (max {args.memory_bandwidth:,.2f}) GB/s - {args.y}: {max_test_tp:,.2f} (max {args.max_throughput:,.2f}) GOPS")
    plt.legend()

    os.makedirs(args.save_dir, exist_ok=True)

    plt.savefig(os.path.join(args.save_dir, f"roofline_{args.name}_{args.x}_{args.y}.svg"))
    print(f"[info] plot saved as roofline_{args.name}_{args.x}_{args.y}.svg")

if __name__ == "__main__":
    main()

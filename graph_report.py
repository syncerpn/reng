import argparse
import os
import json
import csv

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-path", type=str, required=True, help="onnx meta data file path")
    parser.add_argument("--output-file", type=str, help="path to output report")

    args = parser.parse_args()

    return args

def parse_graph_txt(file_name):
    data = []

    with open(file_name, "r") as f:
        data = f.readlines()

    data = [[c for c in d.strip().split(" ") if c] for d in data[2:-1]]
    graph = []
    for name, t, macs, _, mem, _, params, _, ishape, oshape in data:
        d_lt = t
        d_macs = int("".join(macs.split(",")))
        d_mem = int("".join(mem.split(",")))
        d_params = int("".join(params.split(",")))
        d_ishape = list(map(int, ishape.split("x")))
        d_ishape = d_ishape + [1] * (4 - len(d_ishape))
            
        d_oshape = list(map(int, oshape.split("x")))
        d_oshape = d_oshape + [1] * (4 - len(d_oshape))
        
        graph.append([d_lt, d_macs, d_mem, d_params, *d_ishape, *d_oshape])

    return graph


def main():
    args = get_args()

    meta_data = {}

    with open(os.path.join(args.meta_path, "meta.json"), "r") as json_file:
        meta_data = json.load(json_file)

    data_all = []

    for i, model_name in enumerate(meta_data):
        graph_file = os.path.join(args.meta_path, "graph", model_name + ".txt")
        graph = []
        try:
            graph = parse_graph_txt(graph_file)

        except:
            print(f"[warn] model {model_name} graph data incomplete")

        if graph:
            data_all.append([model_name, graph])


    print(f"[info] {len(data_all)} entries")
    write_mode = "w"
    if os.path.exists(args.output_file):
        write_mode = "a"
        print(f"[info] appending to the existing report {args.output_file}")

    with open(args.output_file, write_mode) as file:
        writer = csv.writer(file)
        writer.writerows(data_all)

if __name__ == "__main__":
    main()

import argparse
import os
import json
import architecture

KILO = 1e3
MEGA = 1e6
GIGA = 1e9
TERA = 1e12

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-json", type=str, required=True, help="test json file")
    parser.add_argument("--output-file", type=str, help="path to output report")
    parser.add_argument("--ibytes", type=int, default=4, help="input byte size")
    parser.add_argument("--obytes", type=int, default=4, help="output byte size")
    parser.add_argument("--pbytes", type=int, default=4, help="params byte size")
    parser.add_argument("--fbytes", type=int, default=4, help="fmaps byte size")

    args = parser.parse_args()

    return args

HEADER_DEFAULT = [
    "name", "n_layers", "params", "ops", "mems_iop", "mems_wfmap", "operational_intensity_iop", "operational_intensity_wfmap",
    "fps_f_1000",
    "throughput_f_1000",
]

def main():
    args = get_args()

    meta_data = {}

    with open(args.test_json, "r") as json_file:
        meta_data = json.load(json_file)

    data_all = [HEADER_DEFAULT]

    for model_name in meta_data:
        data = meta_data[model_name]
        
        try:
            d_n_layers = len(architecture.parse(model_name)) - 1
            d_params = data["params"]
            d_ops = data["ops"]
            d_mems_iop = data["mems"]["input"] * args.ibytes + data["mems"]["output"] * args.obytes + data["params"] * args.pbytes
            d_mems_wfmap = d_mems_iop + data["mems"]["fmap"] * args.fbytes
            d_oi_iop = 0 if d_mems_iop == 0 else d_ops / d_mems_iop
            d_oi_wfmap = 0 if d_mems_wfmap == 0 else d_ops / d_mems_wfmap

            d_fpss = data["test"]["fps"]
            d_throughputs = [d_fps * d_ops / GIGA if d_fps != -1 else 0 for d_fps in d_fpss]

            data_row = [model_name, d_n_layers, d_params, d_ops, d_mems_iop, d_mems_wfmap, d_oi_iop, d_oi_wfmap, *d_fpss, *d_throughputs, ]

            if any([d_fps <= 0 for d_fps in d_fpss]):
                print(f"[warn] model {model_name} fps invalid, possibly due to a failed test")
            else:
                data_all.append(data_row)

        except:
            print(f"[warn] model {model_name} test data incomplete: {data}")

    if len(data_all) < 2:
        print(f"[warn] no test data found")
    else:
        print(f"[info] {len(data_all) - 1} entries")
        write_mode = "w"
        if os.path.exists(args.output_file):
            write_mode = "a"
            print(f"[info] appending to the existing report {args.output_file}")
            data_all = data_all[1:]

        with open(args.output_file, write_mode) as file:
            file.write("".join([",".join(list(map(str, d))) + "\n" for d in data_all]))

if __name__ == "__main__":
    main()

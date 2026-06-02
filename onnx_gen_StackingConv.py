import argparse
from architecture import StackingConv
import onnx_generator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l", type=str, required=True, help="number of layers")
    parser.add_argument("--ic", type=str, required=True, help="number of input channels")
    parser.add_argument("--f", type=str, required=True, help="number of features")
    parser.add_argument("--oc", type=str, required=True, help="number of output channels")
    parser.add_argument("--k", type=str, required=True, help="kernel size")

    parser.add_argument("--input", type=str, required=True, help="input dims, should be HxW")

    parser.add_argument("--save-dir", type=str, required=True, help="path to save the generated models")
    parser.add_argument("--no-limit", action='store_true', help="allow generating more than 2GB onnx files")

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    hws = args.input.split(",")
    ls = list(map(int, args.l.split(",")))
    ics = list(map(int, args.ic.split(",")))
    fs = list(map(int, args.f.split(",")))
    ocs = list(map(int, args.oc.split(",")))
    ks = list(map(int, args.k.split(",")))

    onnx_generator.generate(StackingConv, [hws, ls, ics, fs, ocs, ks], args.save_dir, args.no_limit)

if __name__ == "__main__":
    main()
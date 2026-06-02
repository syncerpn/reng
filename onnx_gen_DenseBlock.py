import argparse
from architecture import DenseBlock
import onnx_generator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=str, required=True, help="number of blocks")
    parser.add_argument("--c", type=str, required=True, help="number of channels")
    parser.add_argument("--g", type=str, required=True, help="growth rate factor")

    parser.add_argument("--input", type=str, required=True, help="input dims, should be HxW")

    parser.add_argument("--save-dir", type=str, required=True, help="path to save the generated models")
    parser.add_argument("--no-limit", action='store_true', help="allow generating more than 2GB onnx files")

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    hws = args.input.split(",")
    bs = list(map(int, args.b.split(",")))
    cs = list(map(int, args.c.split(",")))
    gs = list(map(int, args.g.split(",")))

    onnx_generator.generate(DenseBlock, [hws, bs, cs, gs], args.save_dir, args.no_limit)

if __name__ == "__main__":
    main()
import torch
import os
import json
from itertools import product

def generate(ArchClass, options, save_dir, no_limit=False):
    configs = list(product(*options))

    print("[info] number of configs: ", len(configs))

    meta_data = {}

    assert not os.path.exists(save_dir), "[info] saving path already exists"
    print("[info] creating save path: ", save_dir)
    os.makedirs(save_dir)
    os.makedirs(os.path.join(save_dir, "onnx"))

    for config in configs:
        core = ArchClass(config)
        core.eval()

        dummy_input = torch.zeros(core.get_input_shape(), requires_grad=True)
        model_name = f"{core.get_name()}_{core.get_desc()}"

        if not no_limit and core.get_params() >= (31 << 24):
            print(f"[warn] skipped: {model_name} due to file size limit (please generate with --no-limit)")
            continue

        meta_data[model_name] = {"input": core.get_input_shape(), "params": core.get_params(), "ops": core.get_ops(), "mems": core.get_mems()}

        torch.onnx.export(
            core,                # The PyTorch model to be converted
            dummy_input,          # The sample input tensor
            os.path.join(save_dir, "onnx", model_name + ".onnx"),         # The name of the output ONNX file
            export_params=True,   # Store the trained weights inside the model
            opset_version=12,     # ONNX version (update depending on your use case)
            do_constant_folding=False,  # Fold constant nodes for optimization
            training=torch.onnx.TrainingMode.EVAL,
            input_names=['input'],     # The model's input names
            dynamic_axes=None
        )

        print(f"[info] generated: {model_name}")

    with open(os.path.join(save_dir, "meta.json"), "w") as json_file:
        json.dump(meta_data, json_file)
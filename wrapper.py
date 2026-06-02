import torch

# torch function signature copied from torch's official documentation
# if the signature ever changes, these implementations need to be changed as well

def Conv2d(name, arch, input_shape,
    in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
    
    layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
    hi, wi = input_shape
    ci = in_channels
    kh, kw = layer.kernel_size
    sh, sw = layer.stride

    # probably the best way to get the output settings is simple just try it
    _, co, ho, wo = layer.forward(torch.zeros([1, ci, hi, wi])).shape

    arch.ops += 2 * kh * kw * ci * co * ho * wo
    arch.params += kh * kw * ci * co

    if bias:
        arch.ops += co * ho * wo
        arch.params += co

    arch.csv_stats.append([name, hi, wi, kh, kw, ci, co, sh, sw, ])

    arch.mems["fmap"] += co * ho * wo

    return layer

def Linear(name, arch,
    in_features, out_features, bias=True, device=None, dtype=None):

    layer = torch.nn.Linear(in_features, out_features, bias, device, dtype)
    ci = in_features

    # probably the best way to get the output settings is simple just try it
    _, co = layer.forward(torch.zeros([1, ci])).shape

    arch.ops += 2 * ci * co
    arch.params += ci * co

    if bias:
        arch.ops += co
        arch.params += co

    arch.csv_stats.append([name, 1, 1, 1, 1, ci, co, 1, 1, ])

    arch.mems["fmap"] += co

    return layer

def AdaptiveAvgPool2d(arch, input_shape,
    output_size):

    layer = torch.nn.AdaptiveAvgPool2d(output_size)
    ci, hi, wi = input_shape

    # probably the best way to get the output settings is simple just try it
    _, co, ho, wo = layer.forward(torch.zeros([1, ci, hi, wi])).shape

    arch.ops += co * ho * wo * (hi - ho + 1) * (wi - wo + 1)

    arch.mems["fmap"] += co * ho * wo

    return layer

def BatchNorm2d(arch, input_shape,
    num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):

    layer = torch.nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats, device, dtype)
    hi, wi = input_shape
    ci = num_features

    arch.ops += 4 * ci * hi * wi # 1 mul + 1 add + 1 sub + 1 sqrt/div (source: chatgpt)
    arch.params += 2 * ci

    arch.mems["fmap"] += ci * hi * wi

    return layer

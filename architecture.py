import torch
import wrapper

def parse(model_name):
    sep = model_name.find("_")
    arch_name = model_name[:sep]
    str_config = model_name[sep+1:]

    return ARCH_CLASS[arch_name](str_config).get_csv_stats()

CONV_CSV_HEADER = ["layer name", "ifmap height", "ifmap width", "filter height", "filter width", "channels", "num filter", "stride height", "stride width", ]

class ArchitectureTemplate(torch.nn.Module):
    def __init__(self, config=None):
        super(ArchitectureTemplate, self).__init__()
        self.ops = 0
        self.params = 0
        self.mems = {
            "input": 0,
            "output": 0,
            "fmap": 0,
        }
        self.csv_stats = [CONV_CSV_HEADER]

        self.desc = ""
        if config:
            self.desc = "_".join(list(map(str, config)))

        self.input_shape = [0]

    def get_name(self):
        return type(self).__name__

    def get_desc(self):
        return self.desc

    def get_input_shape(self):
        return self.input_shape

    def get_ops(self):
        return self.ops

    def get_params(self):
        return self.params

    def get_mems(self):
        return self.mems

    def get_csv_stats(self):
        return self.csv_stats

class StackingConv(ArchitectureTemplate):
    '''
    input -> 1x1 -> l kxk conv -> kxk conv output
    '''
    def __init__(self, config):
        if type(config) is str:
            hw, l, i, f, o, k = config.split("_")
            config = [hw, int(l), int(i), int(f), int(o), int(k)]
        
        super(StackingConv, self).__init__(config)

        hw, l, i, f, o, k = config
        h, w = list(map(int, hw.split("x")))
        self.input_shape = [1, i, h, w]
        self.mems["input"] += h * w * i
        self.mems["output"] += h * w * o
        self.mems["fmap"] -= self.mems["output"]

        self.conv = torch.nn.ModuleList()

        self.conv.append(wrapper.Conv2d("conv.head.0", self, (h, w), i, i, 1, 1, 0))
        self.conv.append(wrapper.Conv2d("conv.head.1", self, (h, w), i, f, k, 1, k//2))

        for li in range(l):
            self.conv.append(wrapper.Conv2d(f"conv.{li}", self, (h, w), f, f, k, 1, k//2))
        
        # last layer
        self.conv.append(wrapper.Conv2d(f"conv.tail", self, (h, w), f, o, k, 1, k//2))

        for layer in self.conv:
            torch.nn.init.normal_(layer.bias, mean=0.0, std=1.0)
            torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        z = x
        for layer in self.conv[:-1]:
            z = torch.nn.functional.relu(layer(z))
        return self.conv[-1](z)

class MutationConv(ArchitectureTemplate):
    '''
    input shape is never changed
    '''
    def __init__(self, config):
        if type(config) is str:
            hw, l, c, k = config.split("_")
            config = [hw, int(l), int(c), int(k)]

        super(MutationConv, self).__init__(config)

        hw, l, c, k = config
        h, w = list(map(int, hw.split("x")))
        self.input_shape = [1, c, h, w]
        self.mems["input"] += h * w * c
        self.mems["output"] += h * w * c
        self.mems["fmap"] -= self.mems["output"]

        self.conv = torch.nn.ModuleList()

        for li in range(l):
            self.conv.append(wrapper.Conv2d(f"conv.{li}", self, (h, w), c, c, k, 1, k//2))

        for layer in self.conv:
            torch.nn.init.normal_(layer.bias, mean=0.0, std=1.0)
            torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        z = x
        for layer in self.conv:
            z = layer(z)
        return z

class ResidualBlock(ArchitectureTemplate):
    '''
    basic residual block
    '''
    def __init__(self, config):
        if type(config) is str:
            hw, b, c = config.split("_")
            config = [hw, int(b), int(c)]

        super(ResidualBlock, self).__init__(config)

        hw, b, c = config
        h, w = list(map(int, hw.split("x")))
        self.input_shape = [1, c, h, w]
        self.mems["input"] += h * w * c
        self.mems["output"] += h * w * c
        self.mems["fmap"] -= self.mems["output"]

        self.block = torch.nn.ModuleList()
        
        self.head = wrapper.Conv2d(f"conv.head", self, (h, w), c, c, 1, 1, 0)

        for bi in range(b):
            self.block.append(
                torch.nn.Sequential(
                    wrapper.Conv2d(f"conv.{bi}.0", self, (h, w), c, c, 3, 1, 1),
                    torch.nn.ReLU(inplace=True),
                    wrapper.Conv2d(f"conv.{bi}.1", self, (h, w), c, c, 3, 1, 1),
                ),
            )

            # residual add
            self.ops += h * w * c
            self.mems["fmap"] += h * w * c

        torch.nn.init.xavier_uniform_(self.head.weight)
        torch.nn.init.normal_(self.head.bias, mean=0.0, std=1.0)
        for bi in self.block:
            for layer in bi:
                if "weight" in layer.__dict__:
                    torch.nn.init.xavier_uniform_(layer.weight)
                if "bias" in layer.__dict__:
                    torch.nn.init.normal_(layer.bias, mean=0.0, std=1.0)

    def forward(self, x):
        z = torch.nn.functional.relu(self.head(x), inplace=True)
        for bi in self.block:
            y = bi(z)
            z = torch.nn.functional.relu(z + y)
        
        return z

class BottleneckResidualBlock(ArchitectureTemplate):
    '''
    common residual block
    '''
    def __init__(self, config):
        if type(config) is str:
            hw, b, c, r = config.split("_")
            config = [hw, int(b), int(c), int(r)]

        super(BottleneckResidualBlock, self).__init__(config)

        hw, b, c, r = config
        h, w = list(map(int, hw.split("x")))
        self.input_shape = [1, c, h, w]
        self.mems["input"] += h * w * c
        self.mems["output"] += h * w * c
        self.mems["fmap"] -= self.mems["output"]

        self.block = torch.nn.ModuleList()
        
        self.head = wrapper.Conv2d(f"conv.head", self, (h, w), c, c, 1, 1, 0)

        for bi in range(b):
            self.block.append(
                torch.nn.Sequential(
                    wrapper.Conv2d(f"conv.{bi}.0", self, (h, w),    c, c//r, 1, 1, 0),
                    torch.nn.ReLU(inplace=True),
                    wrapper.Conv2d(f"conv.{bi}.1", self, (h, w), c//r, c//r, 3, 1, 1),
                    torch.nn.ReLU(inplace=True),
                    wrapper.Conv2d(f"conv.{bi}.2", self, (h, w), c//r,    c, 1, 1, 0),
                ),
            )

            # residual add
            self.ops += h * w * c
            self.mems["fmap"] += h * w * c

        torch.nn.init.xavier_uniform_(self.head.weight)
        torch.nn.init.normal_(self.head.bias, mean=0.0, std=1.0)
        for bi in self.block:
            for layer in bi:
                if "weight" in layer.__dict__:
                    torch.nn.init.xavier_uniform_(layer.weight)
                if "bias" in layer.__dict__:
                    torch.nn.init.normal_(layer.bias, mean=0.0, std=1.0)

    def forward(self, x):
        z = torch.nn.functional.relu(self.head(x), inplace=True)
        for bi in self.block:
            y = bi(z)
            z = torch.nn.functional.relu(z + y)
        
        return z

class TinyResidualBlock(ArchitectureTemplate):
    '''
    tiny 1x1->1x1 residual block
    '''
    def __init__(self, config):
        if type(config) is str:
            hw, b, c, f = config.split("_")
            config = [hw, int(b), int(c), int(f)]

        super(TinyResidualBlock, self).__init__(config)

        hw, b, c, f = config
        h, w = list(map(int, hw.split("x")))
        self.input_shape = [1, c, h, w]
        self.mems["input"] += h * w * c
        self.mems["output"] += h * w * c
        self.mems["fmap"] -= self.mems["output"]

        self.block = torch.nn.ModuleList()
        
        self.head = wrapper.Conv2d(f"conv.head", self, (h, w), c, c, 1, 1, 0)

        for bi in range(b):
            self.block.append(
                torch.nn.Sequential(
                    wrapper.Conv2d(f"conv.{bi}.0", self, (h, w), c, f, 1, 1, 0),
                    torch.nn.ReLU(inplace=True),
                    wrapper.Conv2d(f"conv.{bi}.1", self, (h, w), f, c, 1, 1, 0),
                ),
            )

            # residual add
            self.ops += h * w * c
            self.mems["fmap"] += h * w * c

        torch.nn.init.xavier_uniform_(self.head.weight)
        torch.nn.init.normal_(self.head.bias, mean=0.0, std=1.0)
        for bi in self.block:
            for layer in bi:
                if "weight" in layer.__dict__:
                    torch.nn.init.xavier_uniform_(layer.weight)
                if "bias" in layer.__dict__:
                    torch.nn.init.normal_(layer.bias, mean=0.0, std=1.0)

    def forward(self, x):
        z = torch.nn.functional.relu(self.head(x), inplace=True)
        for bi in self.block:
            y = bi(z)
            z = torch.nn.functional.relu(z + y)
        
        return z

class StackingLinear(ArchitectureTemplate):
    '''
    common fully connect (gemm)
    '''
    def __init__(self, config):
        if type(config) is str:
            l, ic, f, oc = config.split("_")
            config = [int(l), int(ic), int(f), int(oc)]

        super(StackingLinear, self).__init__(config)

        l, ic, f, oc = config
        self.input_shape = [1, ic]
        self.mems["input"] += ic
        self.mems["output"] += oc
        self.mems["fmap"] -= self.mems["output"]

        self.layers = torch.nn.ModuleList()

        self.layers.append(wrapper.Linear("linear.head", self, ic, f))

        for li in range(l):
            self.layers.append(wrapper.Linear(f"linear.{li}", self, f, f))
        
        self.layers.append(wrapper.Linear("linear.tail", self, f, oc))

        for layer in self.layers:
            torch.nn.init.normal_(layer.bias, mean=0.0, std=1.0)
            torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        z = x
        for layer in self.layers:
            z = layer(z)

        return z

class SqueezeExcitationBlock(ArchitectureTemplate):
    '''
    squeeze and excitation block
    '''
    def __init__(self, config):
        if type(config) is str:
            hw, b, c, r = config.split("_")
            config = [hw, int(b), int(c), int(r)]

        super(SqueezeExcitationBlock, self).__init__(config)

        hw, b, c, r = config
        h, w = list(map(int, hw.split("x")))
        self.input_shape = [1, c, h, w]
        self.mems["input"] += h * w * c
        self.mems["output"] += h * w * c
        self.mems["fmap"] -= self.mems["output"]

        self.block = torch.nn.ModuleList()
        
        self.head = wrapper.Conv2d(f"conv.head", self, (h, w), c, c, 1, 1, 0)

        for bi in range(b):
            self.block.append(
                torch.nn.Sequential(
                    wrapper.AdaptiveAvgPool2d(self, (c, h, w), 1),
                    wrapper.Conv2d(f"excitation.{bi}.0", self, (1, 1), c, c//r, 1, 1, 0, bias=False),
                    torch.nn.ReLU(inplace=True),
                    wrapper.Conv2d(f"excitation.{bi}.1", self, (1, 1), c//r, c, 1, 1, 0, bias=False),
                    torch.nn.Sigmoid(),
                ),
            )

            # residual mul
            self.ops += h * w * c
            self.mems["fmap"] += h * w * c

        for bi in self.block:
            for layer in bi:
                if "weight" in layer.__dict__:
                    torch.nn.init.xavier_uniform_(layer.weight)
                if "bias" in layer.__dict__:
                    torch.nn.init.normal_(layer.bias, mean=0.0, std=1.0)

    def forward(self, x):
        z = self.head(x)
        for bi in self.block:
            y = bi(z)
            z = z * y

        return z

class DenseBlock(ArchitectureTemplate):
    '''
    dense block
    '''
    def __init__(self, config):
        if type(config) is str:
            hw, b, c, g = config.split("_")
            config = [hw, int(b), int(c), int(g)]

        super(DenseBlock, self).__init__(config)

        hw, b, c, g = config
        h, w = list(map(int, hw.split("x")))
        self.input_shape = [1, c, h, w]
        self.mems["input"] += h * w * c
        self.mems["output"] += h * w * (c + b * g)
        self.mems["fmap"] -= self.mems["output"]

        self.block = torch.nn.ModuleList()
        
        self.head = wrapper.Conv2d(f"conv.head", self, (h, w), c, c, 1, 1, 0)

        for bi in range(b):
            self.block.append(
                torch.nn.Sequential(
                    wrapper.BatchNorm2d(self, (h, w), c + bi * g),
                    torch.nn.ReLU(inplace=True),
                    wrapper.Conv2d(f"conv.{bi}", self, (h, w), c + bi * g, g, 3, 1, 1, bias=False),
                )
            )

        for bi in self.block:
            for layer in bi:
                if "weight" in layer.__dict__:
                    torch.nn.init.xavier_uniform_(layer.weight)
                if "bias" in layer.__dict__:
                    torch.nn.init.normal_(layer.bias, mean=0.0, std=1.0)

    def forward(self, x):
        z = self.head(x)
        for bi in self.block:
            y = bi(z)
            z = torch.cat([z, y], dim=1)

        return z


ARCH_CLASS = {
    "StackingConv": StackingConv,
    "MutationConv": MutationConv,
    "TinyResidualBlock": TinyResidualBlock,
    "ResidualBlock": ResidualBlock,
    "BottleneckResidualBlock": BottleneckResidualBlock,
    "StackingLinear": StackingLinear,
    "SqueezeExcitationBlock": SqueezeExcitationBlock,
    "DenseBlock": DenseBlock,
}
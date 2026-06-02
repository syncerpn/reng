"""
postprocess related classes and methods
"""

# pylint: disable=unused-argument,missing-docstring

import logging
import time
import cv2
import numpy as np
import utils

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("postprocess")

class PostProcess:
    def __init__(self):
        self.total = 0
        self.psnr = 0
        self.ssim = 0

        self.timestamps = []

    def start(self):
        self.total = 0
        self.psnr = 0
        self.ssim = 0

        self.timestamps = []

    def _process(self, zs, xs, xps):
        return zs

    def __call__(self, zs, xs, xps):
        s = time.perf_counter()
        rs = self._process(zs, xs, xps)
        e = time.perf_counter()
        self.timestamps.append((s, e))
        
        self.total += len(xs)
        return rs

    def finalize(self, results):
        results["total"] = self.total
        results["postprocess_timing"] = self.timestamps

class PostProcessTorchAny(PostProcess):
    def __init__(self):
        super(PostProcessTorchAny, self).__init__()

    def _process(self, zs, xs, xps) -> list:
        zrgb = utils.transpose_NCHW_to_NHWC(zs)
        zrgb = utils.to_uint8(zrgb)
        rs = utils.unbatch(zrgb)

        return rs

class PostProcessHailo8Any(PostProcess):
    def __init__(self):
        super(PostProcessHailo8Any, self).__init__()

    def _process(self, zs, xs, xps) -> list:
        zrgb = utils.to_uint8(zs)
        rs = utils.unbatch(zrgb)

        return rs

class PostProcessDXM1Any(PostProcess):
    def __init__(self):
        super(PostProcessDXM1Any, self).__init__()

    def _process(self, zs, xs, xps) -> list:
        zrgb = utils.to_uint8(zs)
        rs = utils.unbatch(zrgb)

        return rs
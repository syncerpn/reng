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
log = logging.getLogger("preprocess")

class PreProcess:
    def __init__(self):
        self.timestamps = []

    def start(self):
        self.timestamps = []

    def _process(self, xs: list):
        return xs

    def __call__(self, xs: list):
        s = time.perf_counter()
        xps = self._process(xs)
        e = time.perf_counter()
        self.timestamps.append((s, e))
        return xps

    def finalize(self, results):
        results["preprocess_timing"] = self.timestamps

# ok
class PreProcessUINT8RGBNCHW(PreProcess):
    def __init__(self):
        super(PreProcessUINT8RGBNCHW, self).__init__()

    def _process(self, xs: list) -> np.ndarray:
        xps = utils.tobatch(xs)
        xps = utils.transpose_NHWC_to_NCHW(xps)

        return xps

# ok
class PreProcessUINT8RGBNHWC(PreProcess):
    def __init__(self):
        super(PreProcessUINT8RGBNHWC, self).__init__()

    def _process(self, xs: list) -> np.ndarray:
        xps = utils.tobatch(xs)

        return xps
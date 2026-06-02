"""
abstract backend class
"""


# pylint: disable=unused-argument,missing-docstring
import time

class Backend():
    def __init__(self):
        self.timestamps = []

    def start(self):
        self.timestamps = []

    def version(self):
        raise NotImplementedError("Backend:version")

    def name(self):
        raise NotImplementedError("Backend:name")

    def load(self, model_path, inputs=None, outputs=None):
        raise NotImplementedError("Backend:load")

    def __call__(self, x):
        s = time.perf_counter()
        r = self.predict(x)
        e = time.perf_counter()
        self.timestamps.append((s, e))
        return r

    def predict(self, x):
        raise NotImplementedError("Backend:predict")

    def finalize(self, results):
        results["inference_timing"] = self.timestamps
import backend
import numpy as np
from dx_engine import InferenceEngine

import threading

class BackendDXM1Async(backend.Backend):
    def __init__(self):
        super(BackendDXM1Async, self).__init__()
        self.model = None

    def version(self):
        return "DX-M1-Async"

    def name(self):
        return f"dx_m1-async"

    def load(self, model_path):
        print(f"[INFO] load core from dxnn model: {model_path}")
        self.model = InferenceEngine(model_path)

    def wait_async(self, jobs, outputs, m):
        i = 0
        while i < m:
            try:
                req_id = jobs[i]
                outputs[i] = self.model.Wait(req_id)[0]
                i += 1
            except:
                pass
        
    def predict(self, samples, ids=None):
        m = samples.shape[0]
        jobs = []
        outputs = [None] * m
        wait_threads = threading.Thread(target=self.wait_async, args=(jobs, outputs, m))
        wait_threads.start()

        for i in range(m):
            req_id = self.model.RunAsync(samples[i], i)
            jobs.append(req_id)

        wait_threads.join()
        return outputs

class BackendDXM1Sync(backend.Backend):
    def __init__(self):
        super(BackendDXM1Sync, self).__init__()
        self.model = None

    def version(self):
        return "DX-M1-Sync"

    def name(self):
        return f"dx_m1-sync"

    def load(self, model_path):
        print(f"[INFO] load core from dxnn model: {model_path}")
        self.model = InferenceEngine(model_path)
        
    def predict(self, samples, ids=None):
        outputs = []
        for i in range(samples.shape[0]):
            x = samples[i, ...]
            sim_outputs = self.model.Run(x)
            z = np.array(sim_outputs[0])
            outputs.append(z)
        return outputs
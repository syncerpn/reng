"""
backend for hailo8
"""
import time
import backend
import numpy as np
from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                InputVStreamParams, OutputVStreamParams, FormatType)

class BackendHailo8(backend.Backend):
    def __init__(self):
        super(BackendHailo8, self).__init__()
        self.input_name = ""
        self.output_name = ""
        self.model = None

        self.devices = Device.scan()

    def version(self):
        return "Hailo8"

    def name(self):
        return f"hailo8"

    def load(self, model_path):
        print(f"[INFO] load core from hailo hef model: {model_path}")

        self.model = HEF(model_path)
        
        self.input_name  = self.model.get_input_vstream_infos()[0].name
        self.output_name = self.model.get_output_vstream_infos()[0].name

        self.target = None

        while True:
            try:
                self.target = VDevice(device_ids=self.devices)
                break
            except:
                print(f"[INFO] waiting for device", end="\r")
                time.sleep(5)

        self.network_group = self.target.configure(self.model, ConfigureParams.create_from_hef(self.model, interface=HailoStreamInterface.PCIe))[0]
        self.network_group_params = self.network_group.create_params()

        self.input_vstreams_params = InputVStreamParams.make_from_network_group(self.network_group, quantized=True, format_type=FormatType.UINT8)
        self.output_vstreams_params = OutputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=FormatType.FLOAT32)
        
    # output is np.ndarray, NOT list
    def predict(self, samples, ids = None):
        feed = {self.input_name: samples}
        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                    response = infer_pipeline.infer(feed)

        return response[self.output_name]
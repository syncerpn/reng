"""
dataset related classes and methods
"""

# pylint: disable=unused-argument,missing-docstring

import logging
import sys
import time
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dataset")

class Dataset():
    def __init__(self):
        self.X_list = []
        self.X = {}
        self.last_loaded = -1

    def get_item_count(self):
        return len(self.X_list)

    def get_list(self):
        raise NotImplementedError("Dataset:get_list")

    def load_query_samples(self, sample_list):
        self.X = {}
        for sample in sample_list:
            self.X[sample] = self.get_item(sample)
        self.last_loaded = time.perf_counter()

    def unload_query_samples(self, sample_list):
        if sample_list:
            for sample in sample_list:
                if sample in self.X :
                    del self.X[sample]
        else:
            self.X = {}

    def get_samples(self, id_list):
        x_data  = [self.X[i] for i in id_list]
        return x_data

    def get_item_loc(self, id):
        raise NotImplementedError("Dataset:get_item_loc")
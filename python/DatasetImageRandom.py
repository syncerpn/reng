"""
implementation of criteo dataset
"""

# pylint: disable=unused-argument,missing-docstring

import logging
import os
import utils

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("DatasetImageRandom")

from dataset import Dataset

class DatasetImageRandom(Dataset):
    def __init__(self, shape, n):
        super(DatasetImageRandom, self).__init__()
        self.shape = shape
        self.X_list = list(range(n))

    def get_item(self, i):
        return utils.generate_random_image_with_shape(self.shape)
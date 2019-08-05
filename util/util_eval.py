import torch
import numpy as np


class Feature(object):
    def __init__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class ExtractFeature(Feature):
    # Load pre-trained model
    def __init__(self, model):
        self.model = model
        self.features = []

    def __str__(self):
        print(self.model.__class__.__name__)

    def extract(self, inputs):
        pass

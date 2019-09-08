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
        super().__init__()
        self.model = model
        self.features = []
        self.labels = []
        self.model.eval()

    def __str__(self):
        print(self.model.__class__.__name__)

    def extract(self, inputs, labels):
        features = self.model.extract_features(inputs)
        self.features += list(features.cpu().numpy())
        self.labels += list(labels.cpu().numpy())

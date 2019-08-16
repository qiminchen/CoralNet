import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.utils.data as data


class Dataset(data.Dataset):
    data_root = os.path.join('/path/to/data', 'data')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.49327573, 0.579955, 0.5280543],
                             [0.21500333, 0.22538187, 0.22580335])  # To be changed
    ])

    @classmethod
    def read_bool_status(cls, status_file):
        with open(os.path.join(cls.list_root, status_file)) as f:
            lines = f.read()
        return [x == 'True' for x in lines.split('\n')]

    def __init__(self, opt, mode='train'):
        assert mode in ['train', 'valid']
        self.mode = mode
        self.sets = opt.sets
        if self.sets == 'source':
            list_root = os.path.join('/path/to/status', 'status')
        elif self.sets == 'target':
            list_root = os.path.join('/path/to/status', 'status')
        else:
            raise NotImplementedError("Data sets incorrect!")

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

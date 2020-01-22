import os
import json
from os.path import join
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
import torch.utils.data as data
import boto3
from io import BytesIO
from torch.utils.data.dataloader import default_collate

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(data.Dataset):

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_size = 224

    @classmethod
    def read_bool_status(cls, status_file):
        with open(status_file) as f:
            lines = f.read()
        return [x == 'True' for x in lines.split('\n')]

    def __init__(self, opt, mode='valid', local=False):
        assert mode in ['train', 'valid']
        self.mode = mode
        self.local = local

        # Read sources list
        s3 = boto3.resource('s3', endpoint_url="https://s3.nautilus.optiputer.net")
        bucket = s3.Bucket('qic003')
        self.bucket = bucket

    def __getitem__(self, idx):
        return None

    def __len__(self):
        return 1


def collate_data(batch):
    # images, labels = zip(*batch)
    len_batch = len(batch)
    batch = list(filter(lambda x: x is not None, batch))
    if len_batch > len(batch):
        diff = len_batch - len(batch)
        batch = batch + batch[:diff]
    # return torch.cat(images), torch.cat(labels)
    return default_collate(batch)

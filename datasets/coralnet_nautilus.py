import os
import json
import torch
from os.path import join
import numpy as np
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

    def __init__(self, opt, mode='train'):
        assert mode in ['train', 'valid']
        self.mode = mode

        # Read sources list
        s3 = boto3.resource('s3', endpoint_url="https://s3.nautilus.optiputer.net")
        bucket = s3.Bucket('qic003')

        img_bucket = bucket.Object('status/images_for_training.txt')
        images_list = img_bucket.get()['Body'].read().decode('utf-8')
        images_list = images_list.split('\n')

        is_train_bucket = bucket.Object('status/is_train.txt')
        is_train = is_train_bucket.get()['Body'].read().decode('utf-8')
        is_train = [x == 'True' for x in is_train.split('\n')]

        labels_bucket = bucket.Object('status/labels_mapping.json')
        labels_map = labels_bucket.get()['Body'].read().decode('utf-8')
        labels_map = json.loads(labels_map)
        assert len(images_list) == len(is_train)

        samples = []
        for i, img in enumerate(images_list):
            label = labels_map[img.split('/')[1]]
            item_in_split = ((self.mode == 'train') == is_train[i])
            if item_in_split:
                samples.append((join('coral_crop', img), label))
        self.samples = samples
        self.bucket = bucket

    def __getitem__(self, idx):
        image, label = self.samples[idx]
        image_object = self.bucket.Object(image)
        try:
            img = Image.open(BytesIO(image_object.get()['Body'].read())).convert('RGB')
            img = self.transform(img)
            return img, label
        except Exception:
            return None

    def __len__(self):
        return len(self.samples)


def collate_data(batch):
    # images, labels = zip(*batch)
    len_batch = len(batch)
    batch = list(filter(lambda x:x is not None, batch))
    if len_batch > len(batch):
        diff = len_batch - len(batch)
        batch = batch + batch[:diff]
    # return torch.cat(images), torch.cat(labels)
    return default_collate(batch)

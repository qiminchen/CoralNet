import os
import json
import torch
import boto3
import numpy as np
import torch.utils.data as data
from torchvision import transforms
from PIL import Image, ImageFile
from os.path import join
from io import BytesIO
from util.util_augmentation import Transformer
from torch.utils.data.dataloader import default_collate

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(data.Dataset):

    @classmethod
    def read_bool_status(cls, status_file):
        with open(status_file) as f:
            lines = f.read()
        return [x == 'True' for x in lines.split('\n')]

    def __init__(self, opt, mode='train'):
        assert mode in ['train', 'valid']
        self.mode = mode
        self.transformer = Transformer()

        # Build connection
        s3 = boto3.resource('s3', endpoint_url="https://s3.nautilus.optiputer.net")
        bucket = s3.Bucket('qic003')

        img_bucket = bucket.Object('status/beta_status/images_all.txt')
        images_list = img_bucket.get()['Body'].read().decode('utf-8')
        images_list = images_list.split('\n')

        is_train_bucket = bucket.Object('status/beta_status/is_train.txt')
        is_train = is_train_bucket.get()['Body'].read().decode('utf-8')
        is_train = [x == 'True' for x in is_train.split('\n')]

        labels_bucket = bucket.Object('status/beta_status/labels_mapping.json')
        labels_map = labels_bucket.get()['Body'].read().decode('utf-8')
        labels_map = json.loads(labels_map)
        assert len(images_list) == len(is_train)

        samples = []
        for i, img in enumerate(images_list):
            # img example: 's800/1888/i674838_s800_1888_126_1250_ga0.jpg'
            label = labels_map[img.split('/')[1]]
            ga_type = img.split('/')[-1].split('.')[0].split('_')[-1]
            item_in_split = ((self.mode == 'train') == is_train[i])
            if item_in_split:
                samples.append({
                    'image_path': join('beta_cropped', img),
                    'label': label,
                    'ga_type': ga_type,
                })
        self.samples = samples
        self.bucket = bucket

    def __getitem__(self, idx):
        image = self.samples[idx]
        image_object = self.bucket.Object(image['image_path'])
        try:
            img = Image.open(BytesIO(image_object.get()['Body'].read())).convert('RGB')
            img = self.transformer(img, image['ga_type'])
            return img, image['label']
        except Exception:
            return None

    def __len__(self):
        return len(self.samples)


def collate_data(batch):
    len_batch = len(batch)
    batch = list(filter(lambda x: x is not None, batch))
    if len_batch > len(batch):
        diff = len_batch - len(batch)
        batch = batch + batch[:diff]
    return default_collate(batch)

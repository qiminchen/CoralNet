import os
import json
import torch
import boto3
import random
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
    def get_k(cls, num, rarity):
        if rarity >= 100000:
            return min(num, 100000)
        bounds = [12500, 14285, 16666, 20000, 25000, 33333, 50000, 100000]
        ratio = 8
        for idx, (lower, upper) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
            if lower < rarity <= upper:
                ratio = 8 - idx
        return int(num * ratio)

    def __init__(self, opt, mode='train'):
        assert mode in ['train', 'valid']

        # Build connection
        s3 = boto3.resource('s3', endpoint_url="https://s3.nautilus.optiputer.net")
        bucket = s3.Bucket('qic003')

        img_bucket = bucket.Object('status/beta_status/images_all_ga.json')
        images_list = img_bucket.get()['Body'].read().decode('utf-8')
        images_list = json.load(images_list)

        is_train_bucket = bucket.Object('status/beta_status/is_train_ga.json')
        is_train = is_train_bucket.get()['Body'].read().decode('utf-8')
        is_train = json.load(is_train)

        labels_bucket = bucket.Object('status/beta_status/labels_mapping.json')
        labels_map = labels_bucket.get()['Body'].read().decode('utf-8')
        labels_map = json.loads(labels_map)
        assert len(images_list) == len(is_train)

        rarity_bucket = bucket.Object('status/beta_status/labels_rarity.json')
        labels_rarity = rarity_bucket.get()['Body'].read().decode('utf-8')
        labels_rarity = json.load(labels_rarity)

        # sub-sampling and super-sampling
        target_list = []
        for k, v in images_list.items():
            if mode == 'valid':
                target_list += [i for i in v if is_train[i] == 'False']
            else:
                target_list += random.sample([i for i in v if is_train[i] == 'True'],
                                             k=self.get_k(len(v), labels_rarity[k]))

        # example: "s1136/1397/i910611_s1136_1397_365_2556_ga7.jpg"
        samples = []
        for img in target_list:
            samples.append({
                'path': join('beta_cropped', img.split('_ga')[0] + '.jpg'),
                'label': labels_map[img.split('/')[1]],
                'ga_type': img.split('_')[-1].split('.')[0]
            })

        self.samples = samples
        self.bucket = bucket
        self.transformer = Transformer()

    def __getitem__(self, idx):
        image = self.samples[idx]
        image_object = self.bucket.Object(image['path'])
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

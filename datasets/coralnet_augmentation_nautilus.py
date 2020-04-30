import os
import json
import torch
import boto3
import random
import torch.utils.data as data
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
        if 50000 <= rarity <= 100000:
            return num
        bounds = [12500, 14285, 16666, 20000, 25000, 33333, 50000]
        ratio = 8
        for idx, (lower, upper) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
            if lower <= rarity < upper:
                ratio = 8 - idx
        return int(num * ratio / 8)

    def __init__(self, opt, mode='train'):
        assert mode in ['train', 'valid']
        self.mode = mode

        # Build connection
        s3 = boto3.resource('s3', endpoint_url="https://s3.nautilus.optiputer.net")
        bucket = s3.Bucket('qic003')

        img_bucket = bucket.Object('status/beta_status/images_all_ga.json')
        images_list = img_bucket.get()['Body'].read().decode('utf-8')
        self.images_list = json.loads(images_list)

        is_train_bucket = bucket.Object('status/beta_status/is_train_ga.json')
        is_train = is_train_bucket.get()['Body'].read().decode('utf-8')
        self.is_train = json.loads(is_train)

        labels_bucket = bucket.Object('status/beta_status/labels_mapping.json')
        labels_map = labels_bucket.get()['Body'].read().decode('utf-8')
        self.labels_map = json.loads(labels_map)

        rarity_bucket = bucket.Object('status/beta_status/labels_rarity.json')
        labels_rarity = rarity_bucket.get()['Body'].read().decode('utf-8')
        self.labels_rarity = json.loads(labels_rarity)

        self.bucket = bucket
        self.transformer = Transformer()

    def sampling(self):
        self.samples = []
        # sub-sampling and super-sampling
        target_list = []
        for k, v in self.images_list.items():
            if self.mode == 'valid':
                target_list += [i for i in v if self.is_train[i] == 'False']
            else:
                tl = [i for i in v if self.is_train[i] == 'True']
                target_list += random.sample(tl, k=self.get_k(len(tl), self.labels_rarity[k]))

        # example: "s1136/1397/i910611_s1136_1397_365_2556_ga7.jpg"
        samples = []
        for img in target_list:
            samples.append({
                'path': join('beta_cropped', img.split('_ga')[0] + '.jpg'),
                'label': self.labels_map[img.split('/')[1]],
                'ga_type': img.split('_')[-1].split('.')[0]
            })

        self.samples = samples

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

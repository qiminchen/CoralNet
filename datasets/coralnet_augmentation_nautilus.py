import json
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
    def get_k(cls, rarity):
        bounds = [12500, 14285, 16666, 20000, 25000, 33333, 50000, 100000]
        ratio = 8
        for idx, (lower, upper) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
            if lower <= rarity < upper:
                ratio = 8 - idx
        return ratio

    def __init__(self, opt, mode='train'):
        assert mode in ['train', 'valid']
        self.mode = mode

        # Build connection
        s3 = boto3.resource('s3', endpoint_url="https://s3.nautilus.optiputer.net")
        bucket = s3.Bucket('qic003')

        img_bucket = bucket.Object('status/gamma_status/1275/images_all.txt')
        images_list = img_bucket.get()['Body'].read().decode('utf-8')
        images_list = images_list.split('\n')

        is_train_bucket = bucket.Object('status/gamma_status/1275/is_train.txt')
        is_train = is_train_bucket.get()['Body'].read().decode('utf-8')
        is_train = [x == 'True' for x in is_train.split('\n')]

        labels_bucket = bucket.Object('status/gamma_status/1275/labels_mapping.json')
        labels_map = labels_bucket.get()['Body'].read().decode('utf-8')
        self.labels_map = json.loads(labels_map)

        rarity_bucket = bucket.Object('status/gamma_status/1275/labels_rarity.json')
        labels_rarity = rarity_bucket.get()['Body'].read().decode('utf-8')
        self.labels_rarity = json.loads(labels_rarity)

        images_dict = {}
        for i, img in enumerate(images_list):
            label = img.split('/')[1]
            item_in_split = ((self.mode == 'train') == is_train[i])
            if item_in_split:
                if label not in images_dict:
                    images_dict[label] = [img]
                else:
                    images_dict[label] += [img]

        self.bucket = bucket
        self.transformer = Transformer()
        self.images_dict = images_dict
        self.ga_type = ['ga{}'.format(i) for i in range(1, 8)]

        del images_list, is_train, images_dict

    def sampling(self):

        # sub-sampling and super-sampling
        self.samples = []
        for l, v in self.images_dict.items():
            target_list = []
            if self.mode == 'valid':
                target_list = [(i, 'ga0') for i in v]
            else:
                if self.labels_rarity[l] >= 100000:
                    target_list = [(i, 'ga0') for i in
                                   random.sample(v, k=min(100000, len(v)))]
                else:
                    k = self.get_k(self.labels_rarity[l]) - 1
                    assert 0 <= k <= 7
                    for img in v:
                        ga_type = ['ga0'] + random.sample(self.ga_type, k=k)
                        target_list += [(img, ga) for ga in ga_type]

            self.samples += [{
                'path': join('beta_cropped', i),
                'label': self.labels_map[l],
                'ga_type': ga
            } for i, ga in target_list]

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

import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.utils.data as data


class Dataset(data.Dataset):
    data_root = os.path.join('/mnt/cube/qic003/coral/', 'moorea_labelled')
    list_root = os.path.join('/mnt/cube/qic003/coral/', 'status')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.49327573, 0.579955, 0.5280543],
                             [0.21500333, 0.22538187, 0.22580335])
    ])

    @classmethod
    def read_bool_status(cls, status_file):
        with open(os.path.join(cls.list_root, status_file)) as f:
            lines = f.read()
        return [x == 'True' for x in lines.split('\n')]

    def __init__(self, opt, mode='train'):
        assert mode in ['train', 'valid']
        self.mode = mode

        # Load images path, images label and train-test split
        is_train = self.read_bool_status('is_train.txt')
        with open(os.path.join(self.list_root, 'all_images.txt'), 'r') as f:
            lines = f.read()
        items = lines.split('\n')
        assert len(items) == len(is_train)

        # Pack paths into a dict
        samples = []
        for i, item in enumerate(items):
            item_in_split = ((self.mode == 'train') == is_train[i])
            if item_in_split:
                label = int(item.split('/')[0])
                sample_dict = {'img_path': item, 'label': label}
                samples.append(sample_dict)
        self.samples = samples

    def __getitem__(self, idx):
        image = Image.open(os.path.join(
            self.data_root, self.samples[idx]['img_path'])).convert('RGB')
        image = self.transform(image)
        label = self.samples[idx]['label']
        return image, label

    def __len__(self):
        return len(self.samples)

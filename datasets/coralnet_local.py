import os
import json
import torch
from os.path import join
import numpy as np
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
import torch.utils.data as data

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(data.Dataset):
    root = '/media/data2/qic003/datasets/coral_crop'
    # root = '/mnt/pentagon/qic003/coralnet'
    data_root = os.path.join(root, 'data')
    status_root = os.path.join(root, 'status')

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
        with open(join(self.status_root, 'images_for_training.txt'), 'r') as f:
            line = f.read()
        f.close()
        images_list = line.split('\n')
        # Read labels mapping
        with open(join(self.status_root, 'labels_mapping.json'), 'r') as f:
            labels_map = json.load(f)
        f.close()
        is_train = self.read_bool_status(join(self.status_root, 'is_train.txt'))
        assert len(images_list) == len(is_train)

        samples = []
        for i, img in enumerate(images_list):
            label = labels_map[img.split('/')[1]]
            item_in_split = ((self.mode == 'train') == is_train[i])
            if item_in_split:
                samples.append((join(self.data_root, img), label))
        self.samples = samples

    def __getitem__(self, idx):
        image, label = self.samples[idx]
        img = Image.open(image).convert('RGB')
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)


# def collate_data(batch):
#     images, labels = zip(*batch)
#     return torch.cat(images), torch.cat(labels)

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
    data_root = '/mnt/sda/coral/'
    status_root = os.path.join(data_root, 'status')

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
        with open(join(self.status_root, 'train_sources.txt'), 'r') as f:
            line = f.read()
        f.close()
        sources_list = line.split('\n')
        # Read labels mapping
        with open(join(self.status_root, 'labels_mapping.json'), 'r') as f:
            self.labels_map = json.load(f)
        f.close()
        # Used to remove annotations that does not exist in training labels list
        training_labels = [int(i) for i in list(self.labels_map.keys())]

        annotations = []
        is_train_path = self.status_root + '/status_per_img/is_train'
        for source in sources_list:
            source_path = self.data_root + source + '/images/'
            anns_list = sorted([al for al in os.listdir(source_path) if al.endswith('.anns.json')])
            # Read annotations and is_train of each source
            is_train = self.read_bool_status(join(is_train_path, source + '.txt'))
            for i, anno in enumerate(anns_list):
                anno_in_split = ((self.mode == 'train') == is_train[i])
                img_name = source + '/images/' + anno.split('.')[0] + '.jpg'
                if anno_in_split & os.path.exists(join(self.data_root, img_name)):
                    with open(join(source_path, anno), 'r') as f:
                        anns_json = json.load(f)
                    # Only need images that have more than 10 annotations
                    # Remove annotations that does not exist in training labels list
                    anns_json = [a for a in anns_json if a['label'] in training_labels]
                    if len(anns_json) >= 10:
                        anno_tuple = (img_name, anns_json)
                        annotations.append(anno_tuple)

        # If validation, dataloader shuffle will be off, so need to DETERMINISTICALLY
        # shuffle here to have a bit of every class
        self.annotations = annotations

    def __getitem__(self, idx):
        annotations_loaded = []
        labels_loaded = []
        anno_tuple = self.annotations[idx]
        img = Image.open(join(self.data_root, anno_tuple[0])).convert('RGB')
        # For randomly crop annotations
        random_index = list(np.random.choice(len(anno_tuple[1]), 10, replace=False))
        for rand in random_index:
            anno_detail = anno_tuple[1][rand]
            left = anno_detail['row'] - int(self.img_size / 2)
            upper = anno_detail['col'] - int(self.img_size / 2)
            right = left + self.img_size
            bottom = upper + self.img_size
            cropped_img = img.crop((left, upper, right, bottom))
            cropped_img = self.transform(cropped_img)
            annotations_loaded.append(cropped_img)
            labels_loaded.append(torch.tensor(self.labels_map[str(anno_detail['label'])]))
        return torch.stack(annotations_loaded), torch.stack(labels_loaded)

    def __len__(self):
        return len(self.annotations)


def collate_data(batch):
    images, labels = zip(*batch)
    return torch.cat(images), torch.cat(labels)

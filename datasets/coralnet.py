import os
import json
from os.path import join
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.utils.data as data


class Dataset(data.Dataset):
    data_root = os.path.join('/mnt/sda', 'coral')
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
            labels_map = json.load(f)
        f.close()

        annotations = []
        for source in sources_list:
            annotate_path = join(self.status_root, 'annotations')
            is_train_path = join(self.status_root, 'is_train')
            # Read annotations and is_train of each source
            with open(join(annotate_path, source + '.txt'), 'r') as f:
                line = f.read()
            annos = line.split('\n')
            is_train = self.read_bool_status(join(is_train_path, source + '.txt'))
            assert len(annos) == len(is_train)
            for i, anno in enumerate(annos):
                anno_in_split = ((self.mode == 'train') == is_train[i])
                if anno_in_split:
                    anno_detail = anno.split('_')
                    anno_dict = {
                        'img_path': anno_detail[0] + '.jpg',
                        'row': int(anno_detail[1]),
                        'col': int(anno_detail[2]),
                        'label': int(anno_detail[3]),
                        'class': labels_map[anno_detail[3]],
                    }
                    annotations.append(anno_dict)

        # If validation, dataloader shuffle will be off, so need to DETERMINISTICALLY
        # shuffle here to have a bit of every class
        self.annotations = annotations

    def __getitem__(self, idx):
        anno_dict = self.annotations[idx]
        img = Image.open(anno_dict['img_path'])
        img_width, img_height = img.size
        left = anno_dict['row'] - int(self.img_size / 2)
        upper = anno_dict['col'] - int(self.img_size / 2)
        right = left + self.img_size
        bottom = upper + self.img_size
        # Invalid image if left or upper < 0
        if left < 0 or upper < 0 or right > img_width or bottom > img_height:
            return None
        cropped_img = img.crop((left, upper, right, bottom))
        cropped_img = self.transform(cropped_img)
        anno_loaded = {
            'img': cropped_img,
            'class': anno_dict['class']
        }
        return anno_loaded

    def __len__(self):
        return len(self.annotations)

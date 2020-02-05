import os
import json
import torch
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

    data_root = '/home/qimin/Downloads/evaluation/images'

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    def __init__(self, opt, local=False):
        source = opt.source
        self.local = local

        images_list = os.listdir(os.path.join(self.data_root, source))

        # Pack paths into a dict
        annotations = []
        for i, im in enumerate(images_list):
            im_root = os.path.join(self.data_root, source, im)
            anno_list = os.listdir(im_root)
            anns = [os.path.join(im_root, al) for al in anno_list]
            labels = [int(al.split('_')[2]) for al in anno_list]
            assert len(anns) == len(labels)
            save_dir = os.path.join(opt.logdir, source, 'images')
            anno_dict = {'image_dir': im_root,
                         'anns_path': anns,
                         'anns_labels': labels,
                         'anns_save_dir': os.path.join(save_dir, im + '.anns.npy'),
                         'feat_save_dir': os.path.join(save_dir, im + '.features.npy')}
            annotations.append(anno_dict)
        self.annotations = annotations

    def __getitem__(self, idx):
        image = self.annotations[idx]
        loaded_images = []
        for anno in image['anns_path']:
            img = Image.open(anno).convert('RGB')
            img = self.transform(img)
            loaded_images.append(img)
        image['anns_loaded'] = torch.stack(loaded_images)
        image['anns_labels'] = torch.tensor(image['anns_labels'])
        return image

    def __len__(self):
        return len(self.annotations)

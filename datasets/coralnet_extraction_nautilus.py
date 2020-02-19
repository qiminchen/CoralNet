import os
import torch
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
import torch.utils.data as data
import boto3
from io import BytesIO

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(data.Dataset):

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    def __init__(self, opt, local=False):
        source = opt.source + '/'
        s3 = boto3.client('s3', endpoint_url="https://s3.nautilus.optiputer.net")
        resp = s3.list_objects_v2(Bucket='qic003', Prefix='evaluation/images/' + source, Delimiter='/')
        images_list = [r['Prefix'] for r in resp['CommonPrefixes']]

        # Pack paths into a dict
        annotations = []
        for i, im_root in enumerate(images_list):
            im_name = im_root.split('/')[-2]
            resp = s3.list_objects_v2(Bucket='qic003', Prefix=im_root)
            anns = [r['Key'] for r in resp['Contents']]
            labels = [int(al.split('/')[-1].split('_')[2]) for al in anns]
            assert len(anns) == len(labels)
            anno_dict = {'anns_path': anns,
                         'anns_labels': labels,
                         'anns_save_path': im_name + '.features.anns.npy',
                         'feat_save_path': im_name + '.features.json'}
            annotations.append(anno_dict)
        self.annotations = annotations
        self.s3 = s3

    def __getitem__(self, idx):
        image = self.annotations[idx]
        loaded_images = []
        for anno in image['anns_path']:
            anno_object = self.s3.get_object(Bucket='qic003', Key=anno)
            img = Image.open(BytesIO(anno_object['Body'].read())).convert('RGB')
            img = self.transform(img)
            loaded_images.append(img)
        image['anns_loaded'] = torch.stack(loaded_images)
        image['anns_labels'] = torch.tensor(image['anns_labels'])
        return image

    def __len__(self):
        return len(self.annotations)

import os
import torch
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
import torch.utils.data as data

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(data.Dataset):

    data_root = '/home/qimin/Downloads/evaluation/images'

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    def __init__(self, opt, local=True):
        source = opt.source
        self.local = local

        images_list = os.listdir(os.path.join(self.data_root, source))

        # Pack paths into a dict
        annotations = []
        for i, im in enumerate(images_list):
            im_root = os.path.join(self.data_root, source, im)
            anno_list = os.listdir(im_root)
            if len(anno_list) > 0:
                anns = [os.path.join(im_root, al) for al in anno_list]
                labels = [int(al.split('_')[2]) for al in anno_list]
                anno_loc = [{'row': al.split('_')[-2],
                             'col': al.split('_')[-1].split('.')[0]} for al in anno_list]
                assert len(anns) == len(labels)
                anno_dict = {'image_dir': im_root,
                             'anns_path': anns,
                             'anns_labels': labels,
                             'anno_loc': anno_loc,
                             'anns_save_path': im + '.features.anns.npy',
                             'feat_save_path': im + '.features.json',
                             'anno_loc_path': im + '.anno.loc.json'}
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

import torch
from torchvision import transforms


class Transformer:

    def __init__(self):
        # geometric transformation by label rarity
        self.geometric = {
            'ga0': transforms.RandomHorizontalFlip(p=0),
            'ga1': transforms.RandomHorizontalFlip(p=1),
            'ga2': transforms.RandomRotation((90, 90)),
            'ga3': transforms.RandomRotation((180, 180)),
            'ga4': transforms.RandomRotation((270, 270)),
            'ga5': transforms.Compose([
                transforms.RandomRotation((90, 90)),
                transforms.RandomHorizontalFlip(p=1),
            ]),
            'ga6': transforms.Compose([
                transforms.RandomRotation((180, 180)),
                transforms.RandomHorizontalFlip(p=1),
            ]),
            'ga7': transforms.Compose([
                transforms.RandomRotation((270, 270)),
                transforms.RandomHorizontalFlip(p=1),
            ]),
        }
        # photometric transformation
        self.photometric = transforms.RandomApply([
            transforms.RandomChoice([
                transforms.ColorJitter(brightness=0.2),
                transforms.ColorJitter(contrast=0.3),
                transforms.ColorJitter(saturation=0.5),
                transforms.ColorJitter(hue=0.05),
                transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.5, hue=0.05),
            ])
        ], p=0.5)
        # to tensor and normalize
        self.totensor = transforms.ToTensor()

    def __call__(self, x, types):
        out = self.geometric[types](x)
        out = self.photometric(out)
        out = self.totensor(out)

        return out


def dataset_sampling(dataset, opt, collate_data):

    if opt.augmented:
        dataset['train'].sampling()
        dataset['valid'].sampling()

    dataloaders = {
        'train': torch.utils.data.DataLoader(
            dataset['train'],
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_data,
        ),
        'valid': torch.utils.data.DataLoader(
            dataset['valid'],
            batch_size=opt.batch_size,
            num_workers=opt.workers,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=collate_data,
        ),
    }
    return dataset, dataloaders

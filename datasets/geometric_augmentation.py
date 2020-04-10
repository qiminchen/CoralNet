import os
import sys
import json
import random
import argparse
import numpy as np
from PIL import Image
from os.path import join

parser = argparse.ArgumentParser(description='geometric augmentation')
parser.add_argument('source', type=str, help='original image to be cropped')
parser.add_argument('--data_root', type=str, help='path to beta cropped dir')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

with open('/mnt/sda/coral/beta_status/labels_rarity.json', 'r') as f:
    label_rarity = json.load(f)
labels_id = list(label_rarity.keys())


def geometric_augmentation(source, data_root):
    """
    :param source: source to be augmented
    :param data_root: path to beta cropped dir
    :return: augmented images
    """
    labels = os.listdir(join(data_root, source))
    for label in labels:
        # if label not in training label, no data augmentation needed
        # if label has more than 50,000 patches, no data augmentation needed
        if label not in labels_id or label_rarity[label] > 50000:
            continue

        # check how many augmented images needed by labels rarity
        bounds = [12500, 14285, 16666, 20000, 25000, 33333, 50000]
        num_augmented = 7
        for idx, (lower, upper) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
            if lower < label_rarity[label] <= upper:
                num_augmented = 7 - idx

        # augmentation
        images = os.listdir(join(data_root, source, label))
        for image in images:
            img = np.array(Image.open(join(data_root, source, label, image)))
            # 7 augmented images
            augmented = [np.rot90(img), np.rot90(img, 2), np.rot90(img, 3), np.fliplr(img),
                         np.fliplr(np.rot90(img)), np.fliplr(np.rot90(img, 2)), np.fliplr(np.rot90(img, 3))]
            augmented = random.choices(augmented, k=num_augmented)
            for i, ad in enumerate(augmented):
                Image.fromarray(ad).save(join(data_root, source, label,
                                              image.split('.')[0] + '_ad_' + str(i) + '.jpg'))
    return True


try:
    ok = geometric_augmentation(args.source, args.data_root)
except Exception as error:
    print('source {} cannot be augmented: {}'.format(args.source, error))
    exit(1)

print('[source: {}]: augmentation finished'.format(args.source))

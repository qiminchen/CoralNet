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


def _get_k(rarity):
    # check how many augmented images needed by labels rarity
    if rarity <= 200:
        return 7
    if 200 < rarity <= 400:
        return 5
    bounds = [400, 600, 800, 1000]
    k = 3
    for idx, (lower, upper) in enumerate(zip(bounds[:-1], bounds[1:])):
        if lower < rarity <= upper:
            k = 3 - idx
    return k


def geometric_augmentation(source, data_root):
    """
    :param source: source to be augmented
    :param data_root: path to beta cropped dir
    :return: augmented images
    """
    with open(join('/mnt/sda/features/status', source, 'labels_rarity.json'), 'r') as f:
        labels_rarity = json.load(f)

    with open(join('/mnt/sda/features/status', source, 'test_list.txt'), 'r') as f:
        line = f.read()
    test_list = line.split('\n')

    images = os.listdir(join(data_root, source))
    for img in images:
        # test image should not be augmented
        if img in test_list:
            continue
        patches = os.listdir(join(data_root, source, img))
        for patch in patches:
            label = patch.split('_')[2]
            if label not in labels_rarity or labels_rarity[label] > 1000:
                continue

            image = np.array(Image.open(join(data_root, source, img, patch)))
            k = _get_k(labels_rarity[label])
            augmented = [np.rot90(image), np.rot90(image, 2), np.rot90(image, 3), np.fliplr(image),
                         np.fliplr(np.rot90(image)), np.fliplr(np.rot90(image, 2)), np.fliplr(np.rot90(image, 3))]
            augmented = random.sample(augmented, k=k)
            for i, ai in enumerate(augmented):
                Image.fromarray(ai).save(join(data_root, source, img,
                                              patch.replace('jpg', 'ga'+str(i)+'.jpg')))
            del augmented

    return True


try:
    ok = geometric_augmentation(args.source, args.data_root)
except Exception as error:
    print('source {} cannot be augmented: {}'.format(args.source, error))
    exit(1)

print('[source: {}]: augmentation finished'.format(args.source))

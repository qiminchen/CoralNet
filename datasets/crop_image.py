import os
import sys
import numpy as np
import json
import argparse
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='crop images')
parser.add_argument('image_path', type=str, help='original image to be cropped')
parser.add_argument('--crop_size', type=int, default=224, help='crop size')
parser.add_argument('--save_root', type=str, help='root path for saving cropped images')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)


def crop(image_path, crop_size, save_root):
    # Read image
    img_path = image_path + '.jpg'
    image = Image.open(img_path).convert('RGB')
    # Read annotation json
    anno_path = image_path + '.anns.json'
    with open(anno_path, 'r') as f:
        all_annos = json.load(f)

    source_name = image_path.split('/')[-3]
    assert source_name.startswith('s')

    # Image padding
    padding = crop_size
    image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
    image = Image.fromarray(image, 'RGB')

    # Cropping
    for anno in all_annos:
        left = padding + anno['col'] - int(crop_size / 2)
        upper = padding + anno['row'] - int(crop_size / 2)
        right = left + crop_size
        bottom = upper + crop_size
        cropped_img = image.crop((left, upper, right, bottom))

        # Save cropped images to folders by labels
        save_path = os.path.join(save_root, source_name, str(anno['label']))
        if not os.path.isdir(save_path):
            os.system('mkdir -p ' + save_path)
        save_img_name = image_path.split('/')[-1] + '_' + source_name + '_' + str(anno['label']) + \
                        '_' + str(anno['row']) + '_' + str(anno['col']) + '.jpg'

        cropped_img.save(os.path.join(save_path, save_img_name))


try:
    crop(args.image_path, args.crop_size, args.save_root)
except FileNotFoundError as error:
    print("Image {} not found".format(args.image_path))
    exit(1)

print("Crop {} completed".format(args.image_path))

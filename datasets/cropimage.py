import os
import json
import argparse
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='Crop images')
parser.add_argument('image_path', type=str,
                    help='Original image to be cropped')
args = parser.parse_args()


def crop(image_path):
    img_size = 224
    # Open image
    img_path = image_path + '.jpg'
    image = Image.open(img_path).convert('RGB')
    # Open annotation json
    anno_path = image_path + '.anns.json'
    with open(anno_path, 'r') as f:
        all_annos = json.load(f)

    source_name = image_path.split('/')[-3]
    assert source_name.startswith('s')

    for anno in all_annos:
        left = anno['row'] - int(img_size / 2)
        upper = anno['col'] - int(img_size / 2)
        right = left + img_size
        bottom = upper + img_size
        cropped_img = image.crop((left, upper, right, bottom))

        save_path = '/media/qimin/seagate5tb/coral_crop/' + source_name + '/' + str(anno['label'])
        save_img_name = image_path.split('/')[-1] + '_' + source_name + '_' + str(anno['label']) + \
                        '_' + str(anno['row']) + '_' + str(anno['col']) + '.jpg'

        if not os.path.isdir(save_path):
            os.system('mkdir -p ' + save_path)
        else:
            cropped_img.save(os.path.join(save_path, save_img_name))


try:
    crop(args.image_path)
except FileNotFoundError as error:
    print("Image {} not found".format(args.image_path))
    exit(1)

print("Crop {} completed".format(args.image_path))

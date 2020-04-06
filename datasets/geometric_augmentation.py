import os
import sys
import argparse
import PIL
from PIL import Image

parser = argparse.ArgumentParser(description='geometric augmentation')
parser.add_argument('source', type=str, help='original image to be cropped')
parser.add_argument('--data_root', type=str, help='path to beta cropped dir')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)


def geometric_augmentation(source):
    """
    :param source: source to be augmented
    :return: augmented images
    """
    return True


try:
    ok = geometric_augmentation(args.source)
except Exception as error:
    print('source {} cannot be augmented: {}'.format(args.source, error))
    exit(1)

print('[source: {}]: augmentation finished'.format(args.source))

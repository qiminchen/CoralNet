import os
import json


def train_test_split(source, save_path, root_path='/mnt/sda/coral_v2'):
    """
    This function is used to generate the same train/test split as current backend
    :param source: source to be used for train/test splitting
    :param save_path: path to save the images list and train/test list
    :param root_path: root path to the coralnet_v2 dataset
    :return: images list and train/test list
    """
    all_list = os.listdir(os.path.join(root_path, source, 'images'))
    images_all = list(sorted([al.split('.')[0] for al in all_list if al.endswith('.jpg')]))

    is_train = []
    for img in images_all:
        with open(os.path.join(root_path, source, 'images', img + '.meta.json'), 'r') as f:
            meta = json.load(f)
        is_train.append('True') if meta['in_trainset'] else is_train.append('False')

    # save images_all as images_all.txt
    with open(os.path.join(save_path, source, 'images_all.txt'), 'w') as f:
        f.write('\n'.join(images_all))
    # save is_train as is_train.txt
    with open(os.path.join(save_path, source, 'is_train.txt'), 'w') as f:
        f.write('\n'.join(is_train))

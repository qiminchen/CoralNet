import os
from os.path import join
import json
import csv
from collections import Counter


data_root = '/media/qimin/seagate5tb/coral'
stat_root = '/media/qimin/seagate5tb/stat'
beta_cropped_root = '/media/qimin/samsung1tb/beta_cropped'

invalid_labels = ['Unclear', 'All other', 'Off', 'Fuzz', 'TAPE', 'Unknown', 'Framer', 'off' 'subject',
                  'Water', 'Other', 'Not root', 'Blurry', 'No Data', 'None', 'Dead', 'Others', 'Dots off',
                  'Non-sample', 'white net', 'OUT', 'out of focus', 'Trash', 'yellow net', 'Worms', 'Cage']


def get_sources_meta():
    """
    :return: .csv file including source_code, source_name, #_images, #_annotations, #_labels, label_id
    """

    sources = os.listdir(data_root)
    sources.remove('label_set.json')

    sources_meta = []
    for source in sources:
        with open(join(data_root, source, 'meta.json'), 'r') as f:
            meta = json.load(f)
        images_list = os.listdir(os.path.join(data_root, source, 'images'))
        images_list = [i.split('.')[0] for i in images_list if i.endswith('.jpg')]

        labels = []
        for i in images_list:
            with open(join(data_root, source, 'images', i + '.anns.json'), 'r') as f:
                label = json.load(f)
            labels += [i['label'] for i in label]

        sources_meta.append({'code': source,
                             'name': meta['name'],
                             '#_images': len(images_list),
                             '#_annotations': len(labels),
                             '#_labels': len(set(labels)),
                             'label_id': list(sorted(set(labels)))})
        print('[{}: {} complete]'.format(source, meta['name']))

    keys = sources_meta[0].keys()
    with open(join(stat_root, 'sources_meta.csv'), 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(sources_meta)

    return True


def get_labels_meta():
    """
    :return: .csv file including code, group, duplicate_of, ann_count, ann_count_training, is_verified, id, name,
    #_sources_present
    """

    with open(join(data_root, 'label_set.json')) as f:
        label_meta = json.load(f)
    with open(join(stat_root, 'training_sources.txt')) as f:
        line = f.read()
    training_sources = line.split('\n')

    # step 1 - merge duplicates
    duplicates_meta = [i for i in label_meta if i['duplicate_of'] != 'None']
    for i in duplicates_meta:
        for j in label_meta:
            if j['name'] == i['duplicate_of']:
                j['ann_count'] += i['ann_count']
                i['ann_count'] = 0   # soft copy, 'ann_count' in label_meta will be changed as well

    # step 2 - remove labels with suspicious names
    label_meta = [i for i in label_meta if i['name'] not in invalid_labels]

    # step 3 - compute ann_count_training and #_sources_present
    ann_count_training = []
    sources_present = []
    for source in training_sources:
        labels = os.listdir(join(beta_cropped_root, source))
        sources_present += labels
        for label in labels:
            ann_count_training += len(os.listdir(join(beta_cropped_root, source, label))) * [label]

    ann_count_training = dict(Counter(ann_count_training))
    sources_present = dict(Counter(sources_present))
    for i in range(len(label_meta)):
        label_meta[i]['ann_count_training'] = ann_count_training[str(label_meta[i]['id'])]
        label_meta[i]['#_sources_present'] = sources_present[str(label_meta[i]['id'])]

    keys = label_meta[0].keys()
    with open(join(stat_root, 'labels_meta.csv'), 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(label_meta)

    return True


if __name__ == '__main__':
    ok = get_sources_meta()

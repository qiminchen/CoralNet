import os
from os.path import join
import json
import copy
import csv
from collections import Counter


data_root = '/media/qimin/seagate5tb/coral'
stat_root = '/mnt/sda/coral/beta_status'
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
    duplicates_meta = [i for i in copy.deepcopy(label_meta) if i['duplicate_of'] != 'None']
    for i in duplicates_meta:
        for j in label_meta:
            if j['name'] == i['duplicate_of']:
                j['ann_count'] += i['ann_count']
                i['duplicate_of_id'] = j['id']
    # contains no duplicates
    label_meta = [i for i in label_meta if i['duplicate_of'] == 'None']

    # step 2 - get labels with suspicious names
    # consider lower cases and upper cases and substrings
    contain_substr_from_invalid = [i['name'] for i in label_meta
                                   if any(substr.lower() in i['name'].lower() for substr in invalid_labels)]
    label_removed = [i for i in label_meta if i['name'] in contain_substr_from_invalid]
    # label_meta = [i for i in label_meta if i['name'] not in contain_substr_from_invalid]

    # step 3 - compute ann_count_training and #_sources_present
    duplicates_dict = {i['id']: i['duplicate_of_id'] for i in duplicates_meta}
    ann_count_training = []
    sources_present = []
    for idx, source in enumerate(training_sources):
        anns_list = os.listdir(os.path.join(data_root, source, 'images'))
        anns_list = [i for i in anns_list if i.endswith('.anns.json')]

        labels = []
        for i in anns_list:
            with open(join(data_root, source, 'images', i), 'r') as f:
                label = json.load(f)
            labels += [duplicates_dict[i['label']] if i['label'] in duplicates_dict else i['label'] for i in label]

        ann_count_training += labels
        sources_present += list(set(labels))
        print('[{}: {} complete]'.format(idx + 1, source))

    ann_count_training = dict(Counter(ann_count_training))
    sources_present = dict(Counter(sources_present))
    for i in range(len(label_meta)):
        label_meta[i]['ann_count_in_training'] = ann_count_training[label_meta[i]['id']] if \
            label_meta[i]['id'] in ann_count_training else 0
        label_meta[i]['#_sources_present_in_training'] = sources_present[label_meta[i]['id']] if \
            label_meta[i]['id'] in sources_present else 0

    keys = label_meta[0].keys()
    with open(join(stat_root, 'labels_meta.csv'), 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(label_meta)

    keys = label_removed[0].keys()
    with open(join(stat_root, 'label_removed.csv'), 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(label_removed)

    return True


if __name__ == '__main__':
    ok = get_labels_meta()

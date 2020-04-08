import os
import numpy as np
from os.path import join
import json
import copy
import csv
from collections import Counter


data_root = '/media/qimin/seagate5tb/coral'
stat_root = '/mnt/sda/coral/beta_status'
beta_cropped_root = '/mnt/sda/beta_cropped'

invalid_labels = ['Unclear', 'All other', 'Off', 'Fuzz', 'TAPE', 'Unknown', 'Framer', 'off' 'subject',
                  'Water', 'Other', 'Not root', 'Blurry', 'No Data', 'None', 'Dead', 'Others', 'Dots off',
                  'Non-sample', 'white net', 'OUT', 'Out of focus', 'Trash', 'yellow net', 'Worms', 'Cage',
                  'CREP-T1 Tape/Wand', 'Unknown Invertebrate', 'Unknown Sponge/Tunicate', 'Dead Oyster',
                  'Unknown Invert', 'Serpulid worms', 'Serpulid worms, unidentified', 'Unknown, other',
                  'Trash: Human Origin', 'Out-planted colony', 'Others OT', 'Other benthic', 'unknown whait',
                  'No dead coral', 'unknown dead massive', 'Unknown 1', 'Unknown 2', 'Unknown 3', 'Unknown 4',
                  'Unknown 5', 'Unknown 6', 'Unknown 7', 'Unknown 8', 'Unknown 9', 'Unknown 10', 'Unknown 11',
                  'Unknown 12', 'Unknown 13', 'Unknown 14', 'off subject', 'Dead oysters', 'Wand', 'CRED-Wand',
                  'ARMS-CREP-Unavailable', 'ARMS-CREP-Unclassified/Unknown', 'CRED-Unclassified/Unknown',
                  'CREP-T1 Unclassified/Unknown benthos']


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

    # step 2 - remove labels with suspicious names
    # consider lower cases and upper cases and substrings
    # contain_substr_from_invalid = [i['name'] for i in label_meta
    #                                if any(substr.lower() in i['name'].lower() for substr in invalid_labels)]
    # label_removed = [i for i in label_meta if i['name'] in contain_substr_from_invalid]
    label_meta = [i for i in label_meta if i['name'] not in invalid_labels]

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

    with open(join(stat_root, 'labels_meta.json'), 'w') as output_file:
        json.dump(label_meta, output_file, indent=2)

    with open(join(stat_root, 'duplicates_meta.json'), 'w') as output_file:
        json.dump(duplicates_meta, output_file, indent=2)

    return True


def get_final_labels():
    with open(join(stat_root, 'labels_meta.json')) as f:
        label_meta = json.load(f)
    with open(join(stat_root, 'duplicates_meta.json')) as f:
        duplicate_meta = json.load(f)

    total_anns = sum([i['ann_count_in_training'] for i in label_meta])
    for nbr_source in [2, 3, 4, 5]:
        for nbr_patch in [100, 200, 500, 1000]:
            results = [i['ann_count_in_training'] for i in label_meta
                       if i['ann_count_in_training'] >= nbr_patch and
                       i['#_sources_present_in_training'] >= nbr_source]
            print('[min patches: {}, min sources presented: {}]: {} %, {} / {}, {} / {}'.format(
                nbr_patch, nbr_source, np.round(100*sum(results)/total_anns, 2), sum(results), total_anns,
                len(results), len(label_meta)))

    # append duplicate labels meta
    final_labels = [i for i in label_meta if i['ann_count_in_training'] >= 1000 and
                    i['#_sources_present_in_training'] >= 3]
    duplicate_labels = [i for i in duplicate_meta if i['duplicate_of_id'] in [i['id'] for i in final_labels]]

    # label mapping
    label_dict = {final_labels[i]['id']: i for i in range(len(final_labels))}
    for duplicate in duplicate_labels:
        label_dict[duplicate['id']] = label_dict[duplicate['duplicate_of_id']]

    keys = final_labels[0].keys()
    with open(join(stat_root, 'final_labels_meta.csv'), 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(final_labels)

    with open(join(stat_root, 'final_labels_meta.json'), 'w') as output_file:
        json.dump(final_labels, output_file, indent=2)

    with open(join(stat_root, 'labels_mapping.json'), 'w') as output_file:
        json.dump(label_dict, output_file, indent=2)

    return True


def get_training_images():
    """
    get training images list (augmentation images excluded)
    :return: images_all.txt and is_train.txt
    """
    with open(join(stat_root, 'training_sources.txt')) as f:
        line = f.read()
    training_sources = line.split('\n')
    with open(join(stat_root, 'labels_mapping.json')) as f:
        label_id = json.load(f)
    label_id = [str(i) for i in label_id.keys()]

    # list all images grouped by labels
    images_dict = {}
    for source in training_sources:
        labels = os.listdir(join(beta_cropped_root, source))
        for label in labels:
            if label in label_id:
                lst = [join(source, i) for i in os.listdir(join(beta_cropped_root, source, label))]
                if label not in images_dict:
                    images_dict[label] = lst
                else:
                    images_dict[label] += lst

    # randomly split train and test by 90/10
    images_all = []
    is_train = []
    for key, value in images_dict.items():
        train = ['True'] * len(value)
        rand_idx = np.random.choice(len(value), int(len(value) * 0.1), replace=False)
        for idx in rand_idx:
            train[idx] = 'False'
        is_train += train
        images_all += value

    with open(join(stat_root, 'images_all.txt'), 'w') as f:
        f.write('\n'.join(images_all))
    f.close()
    with open(join(stat_root, 'is_train.txt'), 'w') as f:
        f.write('\n'.join(is_train))
    f.close()

    return True


if __name__ == '__main__':
    # ok = get_labels_meta()
    # ok = get_final_labels()
    ok = get_training_images()

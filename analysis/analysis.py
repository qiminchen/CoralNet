import os
from os.path import join
import json
import csv
from collections import Counter


data_root = '/media/qimin/seagate5tb/coral'
save_root = '/media/qimin/seagate5tb/stat'


def get_sources_meta():
    """
    :return: .csv file including source code, source name, # of images, # of annotations, # of labels, label id
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
                             '# images': len(images_list),
                             '# annotations': len(labels),
                             '# labels': len(set(labels)),
                             'label id': list(sorted(set(labels)))})
        print('[{}: {} complete]'.format(source, meta['name']))

    keys = sources_meta[0].keys()
    with open(join(save_root, 'sources_meta.csv'), 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(sources_meta)

    return True


if __name__ == '__main__':
    ok = get_sources_meta()

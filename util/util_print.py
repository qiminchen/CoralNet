import os
import json
import pickle
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import Image, ImageDraw


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


str_stage = bcolors.OKBLUE + '==>' + bcolors.ENDC
str_verbose = bcolors.OKGREEN + '[Verbose]' + bcolors.ENDC
str_warning = bcolors.WARNING + '[Warning]' + bcolors.ENDC
str_error = bcolors.FAIL + '[Error]' + bcolors.ENDC


with open('/home/qimin/Projects/CoralNet/analysis/label_set.json', 'r') as f:
    all_labels = json.load(f)


def plot_refacc(filepath):
    with open(os.path.join(filepath, 'status.json'), 'r') as f:
        status = json.load(f)
    refacc = status['refacc']
    plt.figure(figsize=(4, 4))
    plt.plot(list(range(len(refacc))), refacc)
    plt.title(filepath.split('/')[-1])
    plt.grid(True, ls='--', lw=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(filepath, 'refacc.png'))


def get_class_name(classes):
    cls = []
    for c in classes:
        for ac in all_labels:
            if ac['id'] == c:
                cls.append(ac['name'])
                break
    return cls


def plot_cm(filename, save_path='/home/qimin/Downloads'):
    """
    :param filename: filename of npz files containing the gt and pred
    :param save_path: path for saving confusion matrix
    :return: confusion matrix
    """

    s = np.load(filename)
    source = filename.split('/')[-2]
    cm = confusion_matrix(s['gt'], s['pred'])
    acc = np.round(np.sum(s['gt'] == s['pred']) * 100 / len(s['gt']), decimals=2)
    each_total = np.sum(cm, axis=1)
    idx = np.argsort(-each_total)
    cm = np.round((cm*100/each_total[:, None]), decimals=1)
    cls = get_class_name(s['cls'])
    cls_nbr = [c + ' [n={}]'.format(str(each_total[i])) for i, c in enumerate(cls)]
    cls_nbr = [cls_nbr[i] for i in idx]
    cls = [cls[i] for i in idx]
    cm_sorted = np.zeros(cm.shape)
    for i in range(cm.shape[0]):
        cm_sorted[i, :] = cm[idx[i], :]
        cm_sorted[i, :] = cm_sorted[i, :][idx]
    df_cm = pd.DataFrame(cm_sorted, index=cls_nbr, columns=cls)
    if len(cls) >= 40:
        plt.figure(figsize=(24, 22))
    elif 20 <= len(cls) < 40:
        plt.figure(figsize=(20, 18))
    else:
        plt.figure(figsize=(12, 10))
    sn.set(font_scale=1)
    sn.heatmap(df_cm, annot=True, cmap="Greens", square=True,
               linewidths=0.1, annot_kws={"size": 10}, fmt='g')  # font size
    plt.title('{}: % of the label that is classified as other labels (acc: {}, n:{}) \n'.format(
        source, acc, len(s['gt'])))
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, source + '.png'))


def images_marking(net, source, image_name):
    """
    :param net: network the classifier is trained on, ex. trained_resnet50
    :param source: source that image belongs to, ex. s294
    :param image_name: image name
    :return: marked image
    """
    image_root = '/mnt/sda/coral'
    feat_root = '/home/qimin/Downloads/evaluation/features'
    cls_root = '/home/qimin/Downloads/evaluation/classifier'

    #
    # Open image
    image = Image.open(os.path.join(image_root, source, 'images', image_name + '.jpg')).convert('RGB')
    #
    # Annotation location
    with open(os.path.join(feat_root, net, source, 'images', image_name + '.anno.loc.json'), 'r') as f:
        anno_loc = json.load(f)
    #
    # Load trained classifier
    with open(os.path.join(cls_root, net, source, 'classifier.pkl'), 'rb') as f:
        clf = pickle.load(f)
    #
    # Load classes and labels
    clf_output = np.load(os.path.join(cls_root, net, source, 'output.npz'), allow_pickle=True)
    classes = clf_output['cls']
    classes_dict = clf_output['cls_dict'].item()
    labels_dict = {v: al['name'] for al in all_labels for k, v in classes_dict.items() if al['id'] == k}
    #
    # Load features and corresponding classes
    with open(os.path.join(feat_root, net, source, 'images', image_name + '.features.json'), 'r') as f:
        x = np.array(json.load(f))
    y = np.load(os.path.join(feat_root, net, source, 'images', image_name + '.features.anns.npy'))
    # Remove annotations that their classes is not in target classes
    invalid_classes = list(set(y).difference(classes))
    invalid_idx = [i for i, v in enumerate(y) if v in invalid_classes]
    x = np.delete(x, invalid_idx, axis=0)
    y = np.delete(y, invalid_idx, axis=0)
    anno_loc = np.delete(np.array(anno_loc), invalid_idx, axis=0)
    y = [classes_dict[i] for i in y]
    est = clf.predict(x)
    acc = np.round(np.sum(y == est) * 100 / len(y))

    # Mark gt and pred on each annotation in image
    d = ImageDraw.Draw(image)
    fills = {'True': (255, 255, 0), 'False': (255, 0, 0)}
    d.text((0, 0), "Accuracy: {}".format(acc), fill=fills['False'])
    for i, loc in enumerate(anno_loc):
        row = int(loc['row'][0])
        col = int(loc['col'][0].split('.')[0])
        d.text((col, row), "X: (gt: {}, est: {})".format(labels_dict[y[i]], labels_dict[est[i]]),
               fill=fills[str(est[i] == y[i])])

    image.save(os.path.join(cls_root, net, source, image_name + '.png'))


def get_label_acc_across_sources(result_root):
    sources = sorted(os.listdir(result_root))
    results = {}
    for s in sources:
        output = np.load(os.path.join(result_root, s, 'output.npz'), allow_pickle=True)
        gt = output['gt']
        pred = output['pred']
        # map the labels back to origin
        cls = {v: k for k, v in output['cls_dict'].item().items()}
        for g, p in zip(gt, pred):
            if cls[g] not in results:
                results[cls[g]] = [cls[p]]
            else:
                results[cls[g]] += [cls[p]]
    acc_over_source = {}
    for k, v in results.items():
        acc = np.sum(np.array([k] * len(v)) == np.array(v)) / len(v)
        acc_over_source[k] = acc

    return acc_over_source

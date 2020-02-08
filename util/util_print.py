import os
import json
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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


with open('/home/qimin/Projects/CoralNet/data_analysis/label_set.json', 'r') as f:
    all_labels = json.load(f)


def plot_refacc(filename):
    with open(filename, 'r') as f:
        status = json.load(f)
    refacc = status['refacc']
    plt.figure(figsize=(4, 4))
    plt.plot(list(range(len(refacc))), refacc)


def get_class_name(classes):
    cls = []
    for c in classes:
        for ac in all_labels:
            if ac['id'] == c:
                cls.append(ac['name'])
                break
    return cls


def plot_cm(filename):
    """
    :param filename: filename of npz files containing the gt and pred
    :return: confusion matrix
    """
    s = np.load(filename)
    source = filename.split('/')[-2]
    cm = confusion_matrix(s['gt'], s['pred'])
    acc = np.round(np.sum(s['gt'] == s['pred']) * 100 / len(s['gt']), decimals=2)
    each_total = np.sum(cm, axis=1)
    idx = np.argsort(-each_total)
    cm = np.round((cm*100/each_total[:,None]), decimals=1)
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
        plt.figure(figsize=(10, 8))
    sn.set(font_scale=1)
    sn.heatmap(df_cm, annot=True, cmap="Greens", square=True,
               linewidths=0.1, annot_kws={"size": 10}, fmt='g')  # font size
    plt.title('{}: % of the label that is classified as other labels (acc: {}, n:{}) \n'.format(
        source, acc, len(s['gt'])))
    plt.tight_layout()
    plt.savefig(os.path.join('/home/qimin/Downloads', source + '.png'))

import os
import csv
import json
import pickle
import numpy as np
import seaborn as sn
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
from PIL import Image, ImageDraw
from scipy import interp
from itertools import cycle
from sklearn.exceptions import UndefinedMetricWarning


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


with open('/mnt/sda/coral/label_set.json', 'r') as f:
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


def get_classification_report(source_path):
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    output = np.load(os.path.join(source_path, 'output.npz'), allow_pickle=True)
    cls = {v: k for k, v in output['cls_dict'].item().items()}
    labels = list(cls.keys())
    labels_name = [i['name'] for l in output['cls'] for i in all_labels if i['id'] == l]
    cls_report = classification_report(output['gt'], output['pred'], output_dict=True,
                                       labels=labels, target_names=labels_name)
    for idx, (k, v) in enumerate(cls_report.items()):
        cls_report[k]['id'] = int(output['cls'][idx])
        if idx == len(cls) - 1:
            break
    return cls_report


def get_labels_classification_report(result_path):
    with open('/mnt/sda/coral/beta_status/1275/final_labels_meta.json', 'r') as fp:
        final_labels_meta = json.load(fp)
    final_labels_meta = {i['id']: i['#_sources_present_in_training'] for i in final_labels_meta}
    skip_keys = ["accuracy", "macro avg", "weighted avg"]
    sources = os.listdir(result_path)
    labels_report = {}
    for s in sources:
        with open(os.path.join(result_path, s, 'classification_report.json'), 'r') as fp:
            cls_report = json.load(fp)
        for k, v in cls_report.items():
            if k not in skip_keys:
                if v['id'] not in labels_report:
                    labels_report[v['id']] = {
                        'name': k,
                        'precision': [v['precision']],
                        'recall': [v['recall']],
                        'f1-score': [v['f1-score']],
                        'support': [v['support']]
                    }
                else:
                    labels_report[v['id']]['precision'] += [v['precision']]
                    labels_report[v['id']]['recall'] += [v['recall']]
                    labels_report[v['id']]['f1-score'] += [v['f1-score']]
                    labels_report[v['id']]['support'] += [v['support']]

    labels_metrics = []
    for k, v in labels_report.items():
        weighted = np.array(v['support']) / np.sum(v['support'])
        tsp = final_labels_meta[int(k)] if int(k) in final_labels_meta else 0
        labels_metrics.append({
            'id': k,
            'name': v['name'],
            'train-source-presented': tsp,
            'test-source-presented': len(v['support']),
            'support': np.sum(v['support']),
            'precision': np.sum(v['precision']*weighted),
            'recall': np.sum(v['recall']*weighted),
            'f1-score': np.sum(v['f1-score']*weighted)
        })
    all_support = [i['support'] for i in labels_metrics]
    all_precision = [i['precision'] for i in labels_metrics]
    all_recall = [i['recall'] for i in labels_metrics]
    all_f1 = [i['f1-score'] for i in labels_metrics]
    overall_weighted = np.array(all_support) / np.sum(all_support)
    labels_metrics.append({
        'id': '-',
        'name': '-',
        'train-source-presented': '-',
        'test-source-presented': '-',
        'support': np.sum(all_support),
        'precision': np.sum(np.array(all_precision) * overall_weighted),
        'recall': np.sum(np.array(all_recall) * overall_weighted),
        'f1-score': np.sum(np.array(all_f1) * overall_weighted)
    })

    keys = labels_metrics[0].keys()
    with open(os.path.join(result_path, 'labels_classification_report.csv'), 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(labels_metrics)


def plot_roc(source_path):
    output = np.load(os.path.join(source_path, 'output.npz'), allow_pickle=True)
    cls = {v: k for k, v in output['cls_dict'].item().items()}
    binarized_gt = label_binarize([cls[i] for i in output['gt']],
                                  classes=output['cls'])
    binarized_pred = label_binarize([cls[i] for i in output['pred']],
                                    classes=output['cls'])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(cls)):
        fpr[i], tpr[i], _ = roc_curve(binarized_gt[:, i], binarized_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(binarized_gt.ravel(), binarized_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(cls))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(cls)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(cls)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(15, 10))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(len(cls)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


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

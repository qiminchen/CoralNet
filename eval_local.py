import os
import sys
import json
import time
import random
import argparse
import numpy as np
import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from util.util_print import str_stage


parser = argparse.ArgumentParser(description='Train Logistic Regression classifier')
parser.add_argument('source', type=str, help='Source to be evaluated')
parser.add_argument('--data_root', type=str, help='Path to the data root')
parser.add_argument('--epochs', type=int, help='Number of epoch for training')
parser.add_argument('--outdir', type=str, help='Output directory')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)


def train_classifier(source, epoch, data_root):
    """
    :param source: source to be evaluated
    :param epoch: number of epoch for training
    :param data_root: data root directory
    :return: gt, pred, classes, classes_dict, clf, stat
    """
    print(str_stage, "Setting up")

    # Read features list and is_train list
    #
    train_list, ref_list, test_list = _get_lists(source, data_root)

    # Identify classes common to both train and test. This will be our labelset for the training.
    #
    classes = _get_classes(source, train_list, ref_list, test_list, data_root)
    with open(os.path.join(data_root, source, 'labels.json'), 'r') as f:
        source_classes = json.load(f)
    classes = list(set(classes).intersection(set(source_classes)))
    classes_dict = {classes[i]: i for i in range(len(classes))}

    # Train a classifier
    #
    start_time = time.time()
    ok, clf, refacc = _do_training(source, train_list, ref_list, epoch, classes, classes_dict, data_root)
    if not ok:
        return {'ok': False, 'runtime': 0, 'refacc': 0, 'acc': 0}
    runtime = time.time() - start_time

    # Evaluate trained classifier
    #
    gt, pred, valacc = _evaluate_classifier(source, clf, test_list, classes, classes_dict, data_root)
    stat = {'ok': True, 'runtime': runtime, 'refacc': refacc, 'acc': valacc}

    return gt, pred, classes, classes_dict, clf, stat


def _do_training(source, train_list, ref_list, epochs, classes, classes_dict, data_root):
    """
    :param source: source to be evaluated
    :param train_list: list of training features filename
    :param ref_list: list of referring features filename
    :param epochs: number of epoch for training
    :param classes: classes to be used
    :param classes_dict: classes dictionary to be used
    :param data_root: data root directory
    :return: True, calibrated classifier and training accuracy
    """

    # Figure out # images per mini-batch and batches per epoch.
    batch_size = min(len(train_list), 200)
    n = int(np.ceil(len(train_list) / float(batch_size)))
    unique_class = list(range(len(classes)))
    print(str_stage, "Start training {} with {} epochs: number of images: {}, number of batch: {}, classes: {}".format(
        source, epochs, len(train_list), n, len(classes)))

    # Load reference data (must hold in memory for the calibration).
    x_ref, y_ref = _load_mini_batch(source, ref_list, classes, classes_dict, data_root)

    # Initialize classifier and refset accuracy list
    clf = SGDClassifier(loss='log', average=True)
    refacc = []
    for epoch in range(epochs):
        random.shuffle(train_list)
        mini_batches = _chunkify(train_list, n)
        for mb in mini_batches:
            x, y = _load_mini_batch(source, mb, classes, classes_dict, data_root)
            clf.partial_fit(x, y, classes=unique_class)
        refacc.append(_acc(y_ref, clf.predict(x_ref)))

    # Calibrate classifier
    clf_calibrated = CalibratedClassifierCV(clf, cv='prefit')
    clf_calibrated.fit(x_ref, y_ref)

    return True, clf, refacc


def _evaluate_classifier(source, clf, test_list, classes, classes_dict, data_root):
    """
    :param source: source to be evaluated
    :param clf: trained classifier
    :param test_list: list of testing features filename
    :param classes: classes to be used
    :param classes_dict: classes dictionary to be used
    :param data_root: data root directory
    :return: gt, pred and evaluation accuracy
    """

    # Figure out # images per mini-batch and batches per epoch.
    batch_size = min(len(test_list), 100)
    n = int(np.ceil(len(test_list) / float(batch_size)))
    print(str_stage, "Testing: batch size: {}, number of batch: {}".format(batch_size, n))

    gt, pred, valacc = [], [], []
    for i in range(n):
        mini_batch = test_list[i*batch_size:(i+1)*batch_size]
        x, y = _load_mini_batch(source, mini_batch, classes, classes_dict, data_root)
        est = clf.predict(x)
        gt.extend(y)
        pred.extend(est)
        valacc.append(_acc(y, est))

    return gt, pred, valacc


def _chunkify(lst, n):
    """
    :param lst: list to be chunkified
    :param n: number of chunks
    :return: list of chunks
    """
    return [lst[i::n] for i in range(n)]


def _load_data(source, xf, yf, classes, data_root):
    """
    :param source: source to be evaluated
    :param xf: image_name.features.json
    :param yf: image_name.features.anns.npy
    :param classes: classes to be used
    :param data_root: data root directory
    :return: loaded features and labels
    """
    with open(os.path.join(data_root, source, 'images', xf), 'r') as f:
        x = json.load(f)
    y = list(np.load(os.path.join(data_root, source, 'images', yf)))

    # Remove samples for which the label is not in classes
    # Check if list of tuple is empty of not
    lot = list(zip(*[(xm, ym) for xm, ym in zip(x, y) if ym in classes]))
    if lot:
        x, y = lot
    else:
        x, y = [], []
    return list(x), list(y)


def _load_mini_batch(source, lst, classes, classes_dict, data_root):
    """
    :param source: source to be evaluated
    :param lst: filename list to be loaded
    :param classes: classes to be used
    :param classes_dict: classes dictionary to be used
    :param data_root: data root directory
    :return: loaded features batch and labels batch
    """
    x_list, y_list = _split(lst)
    x, y = [], []
    for i in range(len(x_list)):
        thisx, thisy = _load_data(source, x_list[i], y_list[i], classes, data_root)
        x.extend(thisx)
        y.extend(thisy)
    y = [classes_dict[i] for i in y]
    return x, y


def _get_classes(source, train_list, ref_list, test_list, data_root):
    """
    :param source: source to be evaluated
    :param train_list: list of training features filename
    :param ref_list: list of referring features filename
    :param test_list: list of testing features filename
    :param data_root: data root directory
    :return: common classes in train, ref and test
    """

    def read(lst):
        lst_classes = []
        for l in lst:
            npy = os.path.join(data_root, source, 'images', l)
            arr = list(np.load(npy))
            lst_classes += arr
        return lst_classes

    _, y_train_list = _split(train_list)
    _, y_ref_list = _split(ref_list)
    _, y_test_list = _split(test_list)
    y_train_classes = read(y_train_list)
    y_ref_classes = read(y_ref_list)
    y_test_classes = read(y_test_list)
    classes = list(set(y_test_classes).intersection(
        set(y_train_classes), set(y_ref_classes)))
    return classes


def _get_lists(source, data_root):
    """
    :param source: source to be evaluated
    :param data_root: data root directory
    :return: training set and testing set filename lists
    """
    with open(os.path.join(data_root, source, 'features_all.txt'), 'r') as file:
        line = file.read()
    file.close()
    features_list = line.split('\n')

    with open(os.path.join(data_root, source, 'is_train.txt'), 'r') as file:
        line = file.read()
    file.close()
    is_train = [x == 'True' for x in line.split('\n')]

    assert len(features_list) == len(is_train)

    # Training set and testing set split and shuffle
    train_list = [features_list[i] for i in range(len(features_list)) if is_train[i] is True]
    test_list = [features_list[i] for i in range(len(features_list)) if is_train[i] is False]
    random.shuffle(train_list)
    random.shuffle(test_list)

    # Make train and ref split. Reference set is here a hold-out part of the train-data portion.
    # Purpose of refset is to 1) know accuracy per epoch and 2) calibrate classifier output scores.
    # We call it 'ref' to disambiguate from the actual validation set of the source.
    ref_list = train_list[:int(len(train_list)*0.1)]
    train_list = list(set(train_list) - set(ref_list))

    return train_list, ref_list, test_list


def _split(lst):
    """
    :param lst: list to be split
    :return: list of feature filenames and label filenames
    """
    x = [l.split(', ')[0] for l in lst]
    y = [l.split(', ')[1] for l in lst]
    return x, y


def _acc(gts, preds):
    """
    :param gts: ground truth label
    :param preds: prediction label
    :return: accuracy
    """
    if len(gts) == 0 or len(preds) == 0:
        raise TypeError('Inputs can not be empty')
    if not len(gts) == len(preds):
        raise ValueError('Input gt and pred must have the same length')

    return float(np.sum(np.array(gts) == np.array(preds).astype(int)) / len(gts))


# gt, pred, cls, status = train_classifier(args.source, args.epochs)
try:
    gt, pred, cls, cls_dict, clf, status = train_classifier(args.source, args.epochs, args.data_root)
except Exception as e:
    gt, pred, cls, cls_dict, clf, status = 0, 0, 0, 0, 0, {'ok': False, 'runtime': 0, 'refacc': 0, 'acc': 0}
    print("Failed to train source: {}, {}".format(args.source, e))
    exit(1)

# Create save dir if not exist
save_dir = os.path.join(args.outdir, args.source)
if not os.path.isdir(save_dir):
    os.system('mkdir -p ' + save_dir)
# Save status to json file
with open(os.path.join(save_dir, 'status.json'), 'w') as f:
    json.dump(status, f)
f.close()
# Save ground truth label and predicted labels to numpy file	# Save trained classifier
np.savez(os.path.join(save_dir, 'output.npz'), gt=gt, pred=pred, cls=cls, cls_dict=cls_dict)
# Save trained classifier
with open(os.path.join(save_dir, 'classifier.pkl'), 'wb') as f:
    pickle.dump(clf, f)
print('{} training completed!'.format(args.source))

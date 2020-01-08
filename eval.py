import os
import sys
import json
import time
import random
import argparse
import numpy as np
import boto3
from io import BytesIO
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from util.util_print import str_stage


parser = argparse.ArgumentParser(description='Train Logistic Regression Classifier')
parser.add_argument('source', type=str, help='Source to be evaluated')
parser.add_argument('--epochs', type=int, help='Number of epoch for training')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)


def train_classifier(source, epoch):
    """
    :param source: source to be evaluated
    :param epoch: number of epoch for training
    :return:
    """
    print(str_stage, "Setting up")

    # Setup aws S3 connection
    #
    s3 = boto3.resource('s3', endpoint_url="https://s3.nautilus.optiputer.net")
    bucket = s3.Bucket('qic003')

    # Read features list and is_train list
    #
    train_list, ref_list, test_list = _get_lists(source, bucket)

    # Identify classes common to both train and test. This will be our labelset for the training.
    #
    classes = _get_classes(source, train_list, ref_list, test_list, bucket)

    # Train a classifier
    #
    start_time = time.time()
    ok, clf, refacc = _do_training(source, train_list, ref_list, epoch, classes, bucket)
    if not ok:
        return {'ok': False, 'runtime': 0, 'refacc': 0, 'acc': 0}
    runtime = time.time() - start_time

    # Evaluate trained classifier
    #
    valacc = _evaluate_classifier(source, clf, test_list, classes, bucket)

    return {'ok': True, 'runtime': runtime, 'refacc': refacc, 'acc': valacc}


def _do_training(source, train_list, ref_list, epochs, classes, bucket):
    """
    :param bucket: S3 bucket
    :param train_list: training features filename list
    :param epochs: number of epoch for training
    :param classes: classes to be used
    :return: True, calibrated classifier and training accuracy
    """

    # Figure out # images per mini-batch and batches per epoch.
    batch_size = min(len(train_list), 200)
    n = int(np.ceil(len(train_list) / float(batch_size)))
    print(str_stage, "Training: batch size: {}, number of batch: {}".format(batch_size, n))
    class_dict = {classes[i]: i for i in range(len(classes))}
    unique_class = list(range(len(classes)))

    # Load reference data (must hold in memory for the calibration).
    x_ref, y_ref = _load_mini_batch(source, ref_list, classes, class_dict, bucket)

    # Initialize classifier and refset accuracy list
    print(str_stage, "Start training classifier")
    clf = SGDClassifier(loss='log', average=True)
    refacc = []
    for epoch in range(epochs):
        random.shuffle(train_list)
        mini_batches = _chunkify(train_list, n)
        for mb in mini_batches:
            x, y = _load_mini_batch(source, mb, classes, class_dict, bucket)
            clf.partial_fit(x, y, classes=unique_class)
        refacc.append(_acc(y_ref, clf.predict(x_ref)))

    # Calibrate classifier
    clf_calibrated = CalibratedClassifierCV(clf, cv='prefit')
    clf_calibrated.fit(x_ref, y_ref)

    return True, clf, refacc


def _evaluate_classifier(source, clf, test_list, classes, bucket):
    """
    :param source: source to be evaluated
    :param clf: trained classifier
    :param test_list: testing features filename list
    :param classes: classes to be used
    :param bucket: S3 bucket
    :return: evaluation accuracy
    """

    # Figure out # images per mini-batch and batches per epoch.
    batch_size = min(len(test_list), 100)
    n = int(np.ceil(len(test_list) / float(batch_size)))
    print(str_stage, "Testing: batch size: {}, number of batch: {}".format(batch_size, n))
    class_dict = {classes[i]: i for i in range(len(classes))}

    print(str_stage, "Start training classifier")
    valacc = []
    for i in range(n):
        mini_batch = test_list[i*batch_size:(i+1)*batch_size]
        x, y = _load_mini_batch(source, mini_batch, classes, class_dict, bucket)
        valacc.append(_acc(y, clf.predict(x)))

    return valacc


def _chunkify(lst, n):
    return [lst[i::n] for i in range(n)]


def _load_data(source, xf, yf, classes, bucket):
    feature_object = bucket.Object('features/' + source + '/images/' + xf)
    x = json.load(BytesIO(feature_object.get()['Body'].read()))
    label_object = bucket.Object('features/' + source + '/images/' + yf)
    y = list(np.load(BytesIO(label_object.get()['Body'].read())))

    # Remove samples for which the label is not in classes
    x, y = zip(*[(xm, ym) for xm, ym in zip(x, y) if ym in classes])
    return list(x), list(y)


def _load_mini_batch(source, lst, classes, class_dict, bucket):
    """
    :param lst: filename list to be loaded
    :return: numpy array features and labels
    """
    x_list, y_list = _split(lst)
    x, y = [], []
    for i in range(len(x_list)):
        thisx, thisy = _load_data(source, x_list[i], y_list[i], classes, bucket)
        x.extend(thisx)
        y.extend(thisy)
    y = [class_dict[i] for i in y]
    return x, y


def _get_classes(source, train_list, ref_list, test_list, bucket):

    def read(lst):
        lst_classes = []
        for l in lst:
            npy = bucket.Object('features/' + source + '/images/' + l)
            arr = list(np.load(BytesIO(npy.get()['Body'].read())))
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


def _get_lists(source, bucket):
    """
    :param source: source to be evaluated
    :param bucket: aws s3 bucket
    :return: training set and testing set filename lists
    """
    features_bucket = bucket.Object('features/' + source + '/features_all.txt')
    features_list = features_bucket.get()['Body'].read().decode('utf-8')
    features_list = features_list.split('\n')

    is_train_bucket = bucket.Object('features/' + source + '/is_train.txt')
    is_train = is_train_bucket.get()['Body'].read().decode('utf-8')
    is_train = [x == 'True' for x in is_train.split('\n')]

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
    x = [l.split(', ')[0] for l in lst]
    y = [l.split(', ')[1] for l in lst]
    return x, y


def _acc(gt, pred):
    """
    :param gt: ground truth label
    :param pred: prediction label
    :return: accuracy
    """
    if len(gt) == 0 or len(pred) == 0:
        raise TypeError('Inputs can not be empty')
    if not len(gt) == len(pred):
        raise ValueError('Input gt and pred must have the same length')

    return float(np.sum(np.array(gt) == np.array(pred).astype(int)) / len(gt))


try:
    status = train_classifier(args.source, args.epochs)
except Exception as e:
    status = {'ok': False, 'runtime': 0, 'refacc': 0, 'acc': 0}
    print("Failed to train source: {0}".format(args.source))
    exit(1)

log_dir = '../qic003/result/eval/' + args.source + '.json'
with open(log_dir, 'w') as f:
    json.dump(status, f)
f.close()
print('{} training completed!'.format(args.source))

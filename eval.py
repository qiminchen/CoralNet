#
import os
import boto
import json
import time
import pickle
import random
import numpy as np
import boto3
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from util.util_print import str_stage


def train_classifier():
    print(str_stage, "Start training classifier")

    # Setup aws S3 connection
    #
    s3 = boto3.resource('s3', endpoint_url="https://s3.nautilus.optiputer.net")
    bucket = s3.Bucket('qic003')

    # Train a classifier
    #
    _do_training(bucket)

    # Evaluate trained classifier
    #
    _evaluate_classifier(bucket)

    return None


def _do_training(bucket):
    """
    :param bucket: S3 bucket
    :return: True, calibrated classifier and training accuracy
    """

    # Read train_features.txt.
    #

    # Make train and ref split. Reference set is here a hold-out part of the train-data portion.
    # Purpose of refset is to 1) know accuracy per epoch and 2) calibrate classifier output scores.
    # We call it 'ref' to disambiguate from the actual validation set of the source.
    #

    # Identify classes common to both train and val. This will be our labelset for the training.
    #

    # Load reference data (must hold in memory for the calibration).
    #

    # Initialize classifier and refset accuracy list
    #

    # Calibrate classifier
    #

    return True


def _evaluate_classifier(bucket):
    pass


def load_features():
    pass


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

    for g in gt:
        if not isinstance(g, int):
            raise TypeError('Input gt must be an array of ints')
    for e in pred:
        if not isinstance(e, int):
            raise TypeError('Input pred must be an array of ints')

    return float(np.sum(gt == pred) / len(gt))

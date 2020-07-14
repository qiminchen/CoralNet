import os
import sys
import json
import time
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from util.util_print import str_stage, plot_cm, plot_trainloss_refacc, get_classification_report
from eval import _get_classes, _load_mini_batch, _chunkify, _acc, _evaluate_classifier


parser = argparse.ArgumentParser(description='LR vs MLP ablation')
parser.add_argument('--source', type=str, help='Source to be evaluated')
parser.add_argument('--data_root', type=str, help='Path to the data root')
parser.add_argument('--epochs', type=int, help='Number of epoch for training')
parser.add_argument('--outdir', type=str, help='Output directory')


def run_training(data_root, source, epochs, outdir):

    source_path = os.path.join('/mnt/sda/features/status', source)
    with open(os.path.join(source_path, 'train_list.txt'), 'r') as file:
        line = file.read()
    train_list = line.split('\n')
    with open(os.path.join(source_path, 'ref_list.txt'), 'r') as file:
        line = file.read()
    ref_list = line.split('\n')
    with open(os.path.join(source_path, 'test_list.txt'), 'r') as file:
        line = file.read()
    test_list = line.split('\n')
    with open(os.path.join(source_path, 'labels.json'), 'r') as f:
        backend_classes = json.load(f)

    hidden_layer_size = [(10, ), (20, ), (50, ), (100, ), (200, ), (200, 100)]
    learning_rate = [1e-3, 1e-4, 1e-5]
    hls_lr = [(i, j) for i in hidden_layer_size for j in learning_rate]

    # different number of images
    for num_string, num_sample in [('im20', 20), ('im30', 30), ('im40', 40),
                                   ('im50', 50), ('im100', 100), ('im200', 200)]:
        print(str_stage, "Training on {}".format(num_string))
        # randomly draw samples
        train_size = round(num_sample * 7 / 8)
        ref_size = round(train_size * 1 / 10)
        test_size = num_sample - train_size
        # 10 runs for LR and MLP with different hyper-parameter setups
        for i in range(10):
            print("==> Run {}".format(i+1))
            train = random.sample(train_list, k=train_size)
            ref = random.sample(ref_list, k=ref_size)
            test = random.sample(test_list, k=test_size)

            classes, _ = _get_classes(source, train, ref, test, data_root)
            classes = list(set(backend_classes).intersection(classes))
            classes_dict = {classes[i]: i for i in range(len(classes))}

            # train LR
            print("====> Training Logistic Regression...")
            start_time = time.time()
            clf = SGDClassifier(loss='log', average=True)
            ok, clf, refacc = _do_training(
                data_root, source, train, ref, epochs, classes, classes_dict, clf
            )
            gt, pred, valacc = _evaluate_classifier(source, clf, test, classes, classes_dict, data_root)
            runtime = start_time - time.time()
            stat = {'ok': True, 'runtime': runtime, 'refacc': refacc, 'acc': np.mean(valacc)}
            log_save_dir = os.path.join(outdir, num_string, source, 'log', str(i+1))
            _visualization(log_save_dir, clf, stat, gt, pred, classes, classes_dict)

            # train MLP with different hyper-parameter setups
            print("====> Training Multi-Layer Perceptron...")
            for hls, lr in hls_lr:
                start_time = time.time()
                clf = MLPClassifier(hidden_layer_sizes=hls, learning_rate_init=lr)
                ok, clf, refacc = _do_training(
                    data_root, source, train, ref, epochs, classes, classes_dict, clf
                )
                gt, pred, valacc = _evaluate_classifier(source, clf, test, classes, classes_dict, data_root)
                runtime = start_time - time.time()
                stat = {'ok': True, 'runtime': runtime, 'refacc': refacc, 'acc': np.mean(valacc)}
                mlp_save_dir = os.path.join(
                    outdir, num_string, source, 'mlp', '_'.join(['_'.join([str(i) for i in hls]),
                                                                 format(lr, '.0e')]), str(i+1)
                )
                _visualization(mlp_save_dir, clf, stat, gt, pred, classes, classes_dict)
                print("[hidden layer size: {}, learning rate: {}] done".format(hls, lr))

    return True


def _do_training(data_root, source, train_list, ref_list, epochs, classes, classes_dict, clf):
    # Figure out # images per mini-batch and batches per epoch.
    batch_size = min(len(train_list), 300)
    n = int(np.ceil(len(train_list) / float(batch_size)))
    unique_class = list(range(len(classes)))

    # Load reference data (must hold in memory for the calibration).
    x_ref, y_ref = _load_mini_batch(source, ref_list, classes, classes_dict, data_root)

    refacc = []
    for epoch in range(epochs):
        random.shuffle(train_list)
        mini_batches = _chunkify(train_list, n)
        for mb in mini_batches:
            x, y = _load_mini_batch(source, mb, classes, classes_dict, data_root)
            clf.partial_fit(x, y, classes=unique_class)
        refacc.append(_acc(y_ref, clf.predict(x_ref)))
    clf_calibrated = CalibratedClassifierCV(clf, cv='prefit')
    clf_calibrated.fit(x_ref, y_ref)

    return True, clf_calibrated, refacc


def _visualization(save_dir, clf, status, gt, pred, cls, cls_dict):
    if not os.path.isdir(save_dir):
        os.system('mkdir -p ' + save_dir)
    # Save status to json file
    with open(os.path.join(save_dir, 'status.json'), 'w') as f:
        json.dump(status, f, indent=2)
    f.close()
    # Save ground truth label and predicted labels to numpy file	# Save trained classifier
    np.savez(os.path.join(save_dir, 'output.npz'), gt=gt, pred=pred, cls=cls, cls_dict=cls_dict)
    with open(os.path.join(save_dir, 'classifier.pkl'), 'wb') as f:
        pickle.dump(clf, f)
    plot_trainloss_refacc(save_dir, 'refacc')
    # Plot confusion matrix
    plot_cm(filename=os.path.join(save_dir, 'output.npz'), save_path=save_dir)
    # Export classification report to .json
    cls_report = get_classification_report(save_dir)
    with open(os.path.join(save_dir, 'classification_report.json'), 'w') as f:
        json.dump(cls_report, f, indent=2)
    # Export classification report to .csv
    cls_report_pd = pd.read_json(os.path.join(save_dir, 'classification_report.json')).T
    cls_report_pd.to_csv(os.path.join(save_dir, 'classification_report.csv'))


if __name__ == "__main__":
    args = parser.parse_args()
    done = run_training(args.data_root, args.source, args.epochs, args.outdir)

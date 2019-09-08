import os
import csv
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


class BaseLogger(object):
    def __init__(self):
        raise NotImplementedError

    def save(self, statistics):
        pass


class CsvLogger(BaseLogger):
    """ loss logger to csv files """
    def __init__(self, opt, filename):
        self.filename = filename
        if opt.resume == 0:
            # create epoch_loss.csv
            with open(self.filename, 'a+', newline='') as cout:
                writer = csv.DictWriter(
                    cout, fieldnames=['phase', 'epoch', 'loss', 'accuracy'])
                writer.writeheader()
            cout.close()

    def save(self, statistics):
        with open(self.filename, 'a+') as csvin:
            writer = csv.writer(csvin)
            writer.writerow(statistics)
        csvin.close()


class ModelLogger(BaseLogger):
    """ model logger to .pt files """
    def __init__(self, filepath):
        self.state_dicts = dict()
        self.filepath = filepath

    def save_state_dict(self, checkpoint, optimizer, filename, additional_values={}):
        # Update checkpoint to the most recent epoch and save
        self.state_dicts['net'] = checkpoint
        self.state_dicts['optimizer'] = optimizer.state_dict()
        for k, v in additional_values.items():
            self.state_dicts[k] = v
        torch.save(self.state_dicts, os.path.join(self.filepath, filename))

    def load_state_dict(self, filepath):
        self.state_dicts = torch.load(filepath)
        additional_values = {k: v for k, v in self.state_dicts.items()
                             if k != 'net'}
        return self.state_dicts, additional_values


class StatisticLogger(BaseLogger):
    """ prediction logger to .npz files """
    def __init__(self, filepath):
        self.filepath = filepath
        self.epoch = 'epoch1_valid'

    def save_metric(self, pred, gt, epoch):
        self.epoch = 'epoch' + str(epoch) + '_valid'
        path = os.path.join(self.filepath, self.epoch)
        if not os.path.isdir(path):
            os.mkdir(path)
        np.savez_compressed(os.path.join(path, 'metric.npz'),
                            pred=pred, gt=gt, epoch=epoch)


class FeatureLogger(BaseLogger):
    """ save features to .npz file"""
    def __init__(self, filepath):
        self.filepath = filepath

    def save_feature(self, features, labels, phase):
        path = os.path.join(self.filepath, phase)
        if not os.path.isdir(path):
            os.mkdir(path)

        pass


def confusion_logger(gt, pred, title, filepath):
    # Classes for Moorea Labelled Dataset
    classes = ['CCA', 'Turf', 'Macroalgae', 'Sand', 'Acropora',
               'Pavona', 'Montipora', 'Pocillopora', 'Porites']
    cm = confusion_matrix(gt, pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Show all ticks
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(filepath, 'confusion_matrix.png'))

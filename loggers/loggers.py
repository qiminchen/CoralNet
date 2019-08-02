import os
import csv
import sys
import torch
import numpy as np


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

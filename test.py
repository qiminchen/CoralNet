# File for testing model: test network forward pass cpu run time

import time
import torch
import models
import datasets
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from options import options_test
from util.util_print import str_stage, str_verbose


###################################################

print(str_stage, "Parsing arguments")
opt = options_test.parse()
# Get all parse done, including subparsers
print(opt)

###################################################

print(str_stage, "Setting device")
device = torch.device('cpu')
print("Device: ", device)

###################################################

print(str_stage, "Setting up models")
model = models.get_model(opt)
state_dicts = torch.load(opt.net_path, map_location=device)
model.load_state_dict(state_dicts['net'])
for param in model.parameters():
    param.requires_grad = False
print("# model parameters: {:,d}".format(
    sum(p.numel() for p in model.parameters() if p.requires_grad)))

###################################################

print(str_stage, "Setting up data loaders")
start_time = time.time()
Dataset = datasets.get_dataset(opt.dataset)
dataset = Dataset(opt, mode='valid')
dataloaders = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.workers,
        pin_memory=True,
        drop_last=True,
        shuffle=False
)
print(str_verbose, "Time spent in data IO initialization: %.2fs" %
      (time.time() - start_time))
print(str_verbose, "# test batches: " + str(len(dataloaders)))

###################################################

print(str_stage, "Start testing CPU run time for forward pass")
data = tqdm(dataloaders, total=len(dataloaders), ncols=120)
# Compute cpu runtime for whole validation set
total_start = time.time()
batch_time = []
for i, (inputs, labels) in enumerate(data):
    inputs = inputs.to(device)
    labels = labels.to(device)
    # Compute cpu runtime per batch
    batch_start = time.time()
    with torch.no_grad():
        outputs = model(inputs)
        _, predicts = torch.max(outputs, 1)
    batch_elapsed = time.time() - batch_start
    batch_time.append(batch_elapsed)
    if i == 99:
        break
total_elapsed = time.time() - total_start
print(str_stage, "CPU run time: maximum {:.0f}m {:.0f}s spent in all batches forwarding".format(
    max(batch_time) // 60, max(batch_time) % 60))
print(str_stage, "CPU run time: average {:.0f}m {:.0f}s spent in all batches forwarding".format(
    np.mean(batch_time) // 60, np.mean(batch_time) % 60))
print(str_stage, "CPU run time: {:.0f}m {:.0f}s spent forwarding {} batches with a batch size of {}".format(
    total_elapsed // 60, total_elapsed % 60, i+1, opt.batch_size))

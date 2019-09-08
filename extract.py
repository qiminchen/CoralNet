# File for evaluating pre-trained CoralNet: extract and save features, train Logistic Regression classifier.

import sys
import os
import time
import torch
import models
import datasets
from tqdm import tqdm
import loggers.loggers as logger
from options import option_extract
from util.util_extract import ExtractFeature
from util.util_print import str_stage, str_verbose, str_warning

###################################################

print(str_stage, "Parsing arguments")
opt = option_extract.parse()
# Get all parse done, including subparsers
print(opt)

###################################################

print(str_stage, "Setting device")
if opt.gpu == '-1':
    device = torch.device('cpu')
    print("# Using CPU")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device('cuda')
    print(device)

###################################################

print(str_stage, "Setting up saving directory")
exprdir = '{}_{}_{}'.format(opt.net, opt.net_version, opt.source)
logdir = os.path.join(opt.logdir, exprdir)
if os.path.isdir(logdir):
    print(
        str_warning, (
            "Will remove Experiment at\n\t%s\n"
            "Do you want to continue? (y/n)"
        ) % logdir
    )
    need_input = True
    while need_input:
        response = input().lower()
        if response in ('y', 'n'):
            need_input = False
    if response == 'n':
        print(str_stage, "User decides to quit")
        sys.exit()
    os.system('rm -rf ' + logdir)
os.system('mkdir -p ' + logdir)
assert os.path.isdir(logdir)
print(str_verbose, "Saving directory set to: %s" % logdir)

###################################################

print(str_stage, "Setting up feature loggers")
feature_logger = logger.FeatureLogger(logdir)

###################################################

print(str_stage, "Setting up models")
model = models.get_model(opt)
state_dicts = torch.load(opt.net_path, map_location=device)
model.load_state_dict(state_dicts['net'])
for param in model.parameters():
    param.requires_grad = False
print("# model parameters: {:,d}".format(
    sum(p.numel() for p in model.parameters() if p.requires_grad)))
model = model.to(device)
extractor = ExtractFeature(model)
# Use model.extract_features(input) to extract features

###################################################

print(str_stage, "Setting up data loaders")
start_time = time.time()
Dataset = datasets.get_dataset(opt.dataset)
dataset = {
    'train': Dataset(opt, mode='train'),
    'valid': Dataset(opt, mode='valid')
}
dataloaders = {
    'train': torch.utils.data.DataLoader(
        dataset['train'],
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True,
        drop_last=True
    ),
    'valid': torch.utils.data.DataLoader(
        dataset['valid'],
        batch_size=opt.batch_size,
        num_workers=opt.workers,
        pin_memory=True,
        drop_last=True,
        shuffle=False
    ),
}
print(str_verbose, "Time spent in data IO initialization: %.2fs" %
      (time.time() - start_time))
print(str_verbose, "# training points: " + str(len(dataset['train'])))
print(str_verbose, "# training batches per epoch: " + str(len(dataloaders['train'])))
print(str_verbose, "# test batches: " + str(len(dataloaders['valid'])))

###################################################

print(str_stage, "Start extracting features")
for phase in ['train', 'valid']:
    data = tqdm(dataloaders[phase], total=len(dataloaders[phase]), ncols=120)
    for inputs, labels in data:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            extractor.extract(inputs, labels)
    feature_logger.save_feature(extractor.features, extractor.labels, phase)
print(str_verbose, "{} feature extraction finished!".format(opt.source))

# File for evaluating pre-trained CoralNet: extract and save features, train Logistic Regression classifier.

import sys
import os
import time
import torch
import models
import datasets
from options import option_extract
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

print(str_stage, "Setting up models")
model = models.get_model(opt)
state_dicts = torch.load(opt.net_path, map_location=device)
model.load_state_dict(state_dicts['net'])
for param in model.parameters():
    param.requires_grad = False
print("# model parameters: {:,d}".format(
    sum(p.numel() for p in model.parameters() if p.requires_grad)))
model = model.to(device)
# Use model.extract_features(input) to extract features

###################################################

print(str_stage, "Setting up data loaders")
start_time = time.time()
Dataset = datasets.get_dataset(opt.dataset)

# File for evaluating pre-trained CoralNet: extract and save features, train Logistic Regression classifier.

import sys
import os
import time
import torch
import models
import datasets
import torch.nn as nn
from tqdm import tqdm
from options import option_extract
from collections import OrderedDict
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
# original saved file with DataParallel
# create new OrderedDict that does not contain `module.
# ref: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
new_state_dict = OrderedDict()
for k, v in state_dicts['net'].items():
    name = k[7:]
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
for param in model.parameters():
    param.requires_grad = False
print("# model parameters: {:,d}".format(
    sum(p.numel() for p in model.parameters() if p.requires_grad)))
# remove _fc layer
model = nn.Sequential(*list(model.children())[:-1])
model = model.to(device)

###################################################

print(str_stage, "Setting up data loaders for source: {}".format(opt.source))
start_time = time.time()
Dataset = datasets.get_dataset(opt.dataset)
dataset = Dataset(opt, local=False)
dataloaders = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True,
        drop_last=True
    )
print(str_verbose, "Time spent in data IO initialization: %.2fs" %
      (time.time() - start_time))
print(str_verbose, "# extracting points: " + str(len(dataset)))
print(str_verbose, "# extracting batches in total: " + str(len(dataloaders)))
exit()

###################################################

print(str_stage, "Start extracting features")
data = tqdm(dataloaders, total=len(dataloaders), ncols=120)
for anns_dict in data:
    inputs = anns_dict['anns_loaded'][0].to(device)
    labels = anns_dict['anns_labels'][0].to(device)
    with torch.no_grad():
        pass
print(str_verbose, "{} feature extraction finished!".format(opt.source))

# Testing model: inference time using CPU and GPU

import os
import time
import torch
import models
import datasets
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
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
if opt.gpu == '-1':
    device = torch.device('cpu')
    print("# Using CPU")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device('cuda')
    print(device)

###################################################

print(str_stage, "Setting up models with image size: {}".format(opt.input_size))
model = models.get_model(opt)
# state_dicts = torch.load(opt.net_path, map_location=device)
# # original saved file with DataParallel
# # create new OrderedDict that does not contain `module`.
# # ref: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
# new_state_dicts = OrderedDict()
# for k, v in state_dicts['net'].items():
#     name = k[7:]
#     # name = k.replace(".module", "")
#     new_state_dicts[name] = v
# model.load_state_dict(new_state_dicts)
for param in model.parameters():
    param.requires_grad = False
print("# model parameters: {:,d}".format(
    sum(p.numel() for p in model.parameters() if p.requires_grad)))
# remove _fc layer
if opt.net == 'resnet':
    model = nn.Sequential(*list(model.children())[:-1])
elif opt.net == 'vgg':
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model = model.to(device)

###################################################

print(str_stage, "Setting up data loaders for source: {}".format(opt.source))
start_time = time.time()
Dataset = datasets.get_dataset(opt.dataset)
dataset = Dataset(opt, local=True)
dataloaders = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=True,
        drop_last=False
    )
print(str_verbose, "Time spent in data IO initialization: %.2fs" %
      (time.time() - start_time))
print(str_verbose, "# extracting points: " + str(len(dataset)))
print(str_verbose, "# extracting batches in total: " + str(len(dataloaders)))

###################################################

print(str_stage, "Start testing inference time")
data = tqdm(dataloaders, total=len(dataloaders), ncols=80)
model.eval()
# Compute inference runtime
inference_time, batch_time = [], []
for i, anns_dict in enumerate(data):
    infer_start = time.time()
    inputs = anns_dict['anns_loaded'][0].to(device)
    with torch.no_grad():
        if opt.net == 'efficientnet':
            outputs = model.extract_features(inputs)
        else:
            outputs = model(inputs).squeeze(-1).squeeze(-1)
    infer_elapsed = time.time() - infer_start
    # Inference time only
    inference_time.append(infer_elapsed)
    # Inference time + data loading time
    batch_time.append(torch.sum(anns_dict['time']).item() + infer_elapsed)
    torch.cuda.empty_cache()
print(str_stage, "Maximum inference time: {}s".format(np.round(max(inference_time), 4)))
print(str_stage, "Average inference time (data loading time excluded): {} ± {} s/per {} image".format(
    np.round(np.mean(inference_time), 4), np.round(np.std(inference_time), 4), opt.batch_size))
print(str_stage, "Average batch time (data loading time included): {} ± {} s/per {} image".format(
    np.round(np.mean(batch_time), 4), np.round(np.std(batch_time), 4), opt.batch_size))

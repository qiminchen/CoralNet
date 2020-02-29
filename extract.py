# File for evaluating pre-trained CoralNet: extract and save features, train Logistic Regression classifier.

import sys
import os
import gc
import time
import json
import torch
import models
import datasets
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from options import option_extract
from collections import OrderedDict
from util.util_print import str_stage, str_verbose

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

print(str_stage, "Setting up models")
model = models.get_model(opt)
state_dicts = torch.load(opt.net_path, map_location=device)
# original saved file with DataParallel
# create new OrderedDict that does not contain `module`.
# ref: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
new_state_dicts = OrderedDict()
for k, v in state_dicts['net'].items():
    name = k[7:]
    # name = k.replace(".module", "")
    new_state_dicts[name] = v
model.load_state_dict(new_state_dicts)
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

print(str_stage, "Start extracting features")
data = tqdm(dataloaders, total=len(dataloaders), ncols=80)
model.eval()
save_dir = os.path.join(opt.logdir, opt.source, 'images')
if not os.path.isdir(save_dir):
    os.system('mkdir -p ' + save_dir)
for i, anns_dict in enumerate(data):
    inputs = anns_dict['anns_loaded'][0].to(device)
    labels = anns_dict['anns_labels'][0].to(device)
    with torch.no_grad():
        if opt.net == 'efficientnet':
            outputs = model.extract_features(inputs)
        else:
            outputs = model(inputs).squeeze(-1).squeeze(-1)
    # save features, corresponding labels and annotation location info
    outputs = outputs.detach().cpu().tolist()
    labels = labels.detach().cpu().numpy()
    with open(os.path.join(save_dir, anns_dict['feat_save_path'][0]), 'w') as fp:
        json.dump(outputs, fp)
    fp.close()
    np.save(os.path.join(save_dir, anns_dict['anns_save_path'][0]), labels)
    with open(os.path.join(save_dir, anns_dict['anno_loc_path'][0]), 'w') as fp:
        json.dump(anns_dict['anno_loc'], fp)
    fp.close()
    del inputs, outputs, labels
    gc.collect()
print(str_verbose, "{} feature extraction finished!".format(opt.source))
# exit()
###################################################

# print(str_stage, "Start creating features_all.txt and is_train.txt")
# status_dir = os.path.join(opt.logdir, opt.source)
# # create features_all.txt
# features_list = list(sorted(os.listdir(os.path.join(status_dir, 'images'))))
# fl = [f for f in features_list if f.endswith('.features.json')]
# fal = [f for f in features_list if f.endswith('.anns.npy')]
# assert len(fl) == len(fal)
# flfal = [fl[i]+', '+fal[i] for i in range(len(fl))]
# with open(os.path.join(status_dir, 'features_all.txt'), 'w') as f:
#     f.write('\n'.join(flfal))
# f.close()
# # create is_train.txt
# nbr = len(flfal)
# is_train = np.array([1]*nbr)
# radn = np.random.choice(nbr, int(nbr*0.125), replace=False)
# is_train[radn] = 0
# is_train = list(is_train)
# is_train = [str(i == 1) for i in is_train]
# assert len(is_train) == nbr
# with open(os.path.join(status_dir, 'is_train.txt'), 'w') as f:
#     f.write('\n'.join(is_train))
# f.close()
# print(str_verbose, "features_all.txt and is_train.txt created")

# File for testing model: test network forward pass cpu run time

import time
import torch
import models
import datasets
from tqdm import tqdm
from options import options_test
from util.util_print import str_error, str_stage, str_verbose, str_warning


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
model.load_state_dict(state_dicts)
for param in model.parameters():
    param.requires_grad = False

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
print(str_verbose, "# test batches: " + str(len(dataloaders['valid'])))

###################################################

print(str_stage, "Start testing CPU run time")
data = tqdm(dataloaders, total=len(dataloaders), ncols=120)
for inputs, labels in enumerate(data):
    pass

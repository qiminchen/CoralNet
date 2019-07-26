import sys
import os
import csv
import time
import torch
import pandas as pd
from tqdm import tqdm
from options import options_train
from datasets import coralnet
from models import resnet50
from util.util_print import str_error, str_stage, str_verbose, str_warning


###################################################

print(str_stage, "Parsing arguments")
opt, unique_opt_params = options_train.parse()
# Get all parse done, including subparsers
print(opt)

###################################################

print(str_stage, "Setting device")
if opt.gpu == '-1':
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')

###################################################

print(str_stage, "Setting up logging directory")
exprdir = '{}_{}_{}'.format(opt.net, opt.dataset, opt.lr)
logdir = os.path.join(opt.logdir, exprdir, str(opt.expr_id))

if opt.resume == 0:
    if os.path.isdir(logdir):
        if opt.expr_id <= 0:
            print(
                str_warning, (
                    "Will remove Experiment %d at\n\t%s\n"
                    "Do you want to continue? (y/n)"
                ) % (opt.expr_id, logdir)
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
        else:
            raise ValueError(str_error +
                             " Refuse to remove positive expr_id")
    os.system('mkdir -p ' + logdir)
else:
    assert os.path.isdir(logdir)
    opt_f_old = os.path.join(logdir, 'opt.pt')
    opt = options_train.overwrite(opt, opt_f_old, unique_opt_params)

# Save opt
torch.save(vars(opt), os.path.join(logdir, 'opt.pt'))
with open(os.path.join(logdir, 'opt.txt'), 'w') as fout:
    for k, v in vars(opt).items():
        fout.write('%20s\t%-20s\n' % (k, v))

opt.full_logdir = logdir
print(str_verbose, "Logging directory set to: %s" % logdir)

###################################################

print(str_stage, "Setting up models")
if opt.net == 'resnet50':
    model = resnet50.resnet50(opt.pretrained)
print("# model parameters: {:,d}".format(
    sum(p.numel() for p in model.parameters() if p.requires_grad)))
model = model.to(device)

initial_epoch = 1
if opt.resume != 0:
    if opt.resume == -1:
        net_filename = os.path.join(logdir, 'checkpoint.pt')
    elif opt.resume == -2:
        net_filename = os.path.join(logdir, 'best.pt')
    else:
        raise NotImplementedError(opt.resume)
    if not os.path.isfile(net_filename):
        print(str_warning, ("Network file not found for opt.resume=%d. "
                            "Starting from scratch") % opt.resume)
    else:
        additional_values = model.load_state_dict(net_filename, load_optimizer='auto')
        try:
            initial_epoch += additional_values['epoch']
        except KeyError as err:
            # Old saved model does not have epoch as additional values
            epoch_loss_csv = os.path.join(logdir, 'epoch_loss.csv')
            if opt.resume == -1:
                try:
                    initial_epoch += pd.read_csv(epoch_loss_csv)['epoch'].max()
                except pd.errors.ParserError:
                    with open(epoch_loss_csv, 'r') as f:
                        lines = f.readlines()
                    initial_epoch += max([int(l.split(',')[0]) for l in lines[1:]])
            else:
                initial_epoch += opt.resume

###################################################

print(str_stage, "Setting up data loaders")
start_time = time.time()
dataset = {
    'train': coralnet.Dataset(opt, mode='train'),
    'valid': coralnet.Dataset(opt, mode='valid')
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
print(str_verbose, "# training points: " + str(len(dataset_train)))
print(str_verbose, "# training batches per epoch: " + str(len(dataloaders['train'])))
print(str_verbose, "# test batches: " + str(len(dataloaders['valid'])))

###################################################

print(str_stage, "Setting up optimizer")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), opt.lr, betas=(opt.adam_beta1, opt.adam_beta2),
                             weight_decay=opt.wdecay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrdecaystep,
                                               gamma=opt.lrdecay)

###################################################

print(str_stage, "Start training")
assert opt.epoch > 0

while initial_epoch <= opt.epoch:
    epoch_loss = 0
    for phase in ['train', 'valid']:
        if phase == 'train':
            lr_scheduler.step()
            model.train()
        else:
            model.eval()
        # Batch loss and batch accuracy
        running_loss = 0.0
        running_accuracy = 0
        # Progress bar
        data = tqdm(dataloaders[phase], desc="Loss: ", total=len(dataloaders[phase]))
        for inputs, labels in data:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, predicts = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                # backward and optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_accuracy += torch.sum(predicts == labels.data)
            # updating progress bar
            data.set_description("{} {}/{}: Loss: {}, Acc: {}".format(
                phase, initial_epoch, opt.epoch, running_loss / len(dataset[phase]),
                running_accuracy / len(dataset[phase])))
        # Save every epoch loss into .csv file
    initial_epoch += 1

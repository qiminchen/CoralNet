import sys
import os
import time
import copy
import torch
import pandas as pd
from tqdm import tqdm
from options import options_train
from datasets import coralnet
import models
import loggers.loggers as logger
import util.util_metric as util_metric
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
if opt.source is not None:
    exprdir = '{}_'.format(opt.source) + exprdir
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

print(str_stage, "Setting up loggers")
csv_logger = logger.CsvLogger(opt, os.path.join(logdir, 'epoch_loss.csv'))
metric_logger = logger.StatisticLogger(logdir)
model_logger = logger.ModelLogger(logdir)

###################################################

print(str_stage, "Setting up models")
model = models.get_model(opt)
print("# model parameters: {:,d}".format(
    sum(p.numel() for p in model.parameters() if p.requires_grad)))
# model = model.to(device)

initial_epoch = 1
if opt.resume == 0:
    checkpoint = copy.deepcopy(model.state_dict())
    best = copy.deepcopy(model.state_dict())
    model_logger.save_state_dict(checkpoint, filename='checkpoint.pt',
                                 additional_values={'epoch': initial_epoch})
    model_logger.save_state_dict(best, filename='best.pt',
                                 additional_values={'epoch': initial_epoch})
else:
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
        checkpoint, additional_values = model_logger.load_state_dict(net_filename)
        model.load_state_dict(checkpoint)
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
print(str_verbose, "# training points: " + str(len(dataset['train'])))
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
best_accuracy = 0.0
running_loss = 0.0
running_accuracy = 0

while initial_epoch <= opt.epoch:
    for phase in ['train', 'valid']:
        if phase == 'train':
            lr_scheduler.step()
            model.train()
        else:
            model.eval()
        # Loss utility
        metric_util = util_metric.MetricUtil()
        # Progress bar
        data = tqdm(dataloaders[phase], desc="Loss: ", total=len(dataloaders[phase]))
        for i, (inputs, labels) in enumerate(data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            running_loss, running_accuracy = metric_util.compute_metric(
                model, phase, inputs, labels, criterion, optimizer, i+1
            )
            # updating progress bar
            data.set_description("{} {}/{}: Loss: {}, Acc: {}".format(
                phase, initial_epoch, opt.epoch, running_loss, running_accuracy))
        # Save every epoch loss into .csv file
        csv_logger.save([phase, initial_epoch, running_loss, running_accuracy])
        # Save best model if exist
        if phase == 'valid' and running_accuracy > best_accuracy:
            best_accuracy = running_accuracy
            best = copy.deepcopy(model.state_dict())
            model_logger.save_state_dict(best, filename='best.pt',
                                         additional_values={'epoch': initial_epoch})
        # Save validation ground truth and prediction
        if phase == 'valid':
            metric_logger.save_metric(metric_util.pred, metric_util.gt, initial_epoch)
    # save most recent model as checkpoint
    checkpoint = copy.deepcopy(model.state_dict())
    model_logger.save_state_dict(checkpoint, filename='checkpoint.pt',
                                 additional_values={'epoch': initial_epoch})
    initial_epoch += 1

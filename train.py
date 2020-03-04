import sys
import os
import time
import copy
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from options import options_train
import datasets
import models
import loggers.loggers as logger
import util.util_metric as util_metric
from datasets.coralnet_nautilus import collate_data
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
elif opt.gpu == '-2':
    device = torch.device('cuda')
    print("Use {} GPUs".format(torch.cuda.device_count()))
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device('cuda')
    print(device)

###################################################

print(str_stage, "Setting up logging directory")
exprdir = '{}_{}_{}_{}'.format(opt.net, opt.net_version,
                               opt.dataset, opt.lr)
exprdir += ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
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
if opt.gpu == '-2':
    model = nn.DataParallel(model)
model = model.to(device)

###################################################

print(str_stage, "Setting up optimizer")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum,
                            weight_decay=opt.wdecay)
# optimizer = torch.optim.Adam(model.parameters(), opt.lr, betas=(opt.adam_beta1, opt.adam_beta2),
#                              weight_decay=opt.wdecay)

###################################################

print(str_stage, "Setting up data loaders and learning rate scheduler")
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
        drop_last=True,
        collate_fn=collate_data,
    ),
    'valid': torch.utils.data.DataLoader(
        dataset['valid'],
        batch_size=opt.batch_size,
        num_workers=opt.workers,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
        collate_fn=collate_data,
    ),
}
print(str_verbose, "Time spent in data IO initialization: %.2fs" %
      (time.time() - start_time))
print(str_verbose, "# training points: " + str(len(dataset['train'])))
print(str_verbose, "# training batches per epoch: " + str(len(dataloaders['train'])))
print(str_verbose, "# test batches: " + str(len(dataloaders['valid'])))

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.max_lr, epochs=opt.epoch,
                                                   steps_per_epoch=len(dataloaders['train']))

###################################################

print(str_stage, "Resuming model information")
initial_epoch = 1
if opt.resume == 0:
    checkpoint = copy.deepcopy(model.state_dict())
    best = copy.deepcopy(model.state_dict())
    model_logger.save_state_dict(checkpoint, optimizer, lr_scheduler, filename='checkpoint.pt',
                                 additional_values={'epoch': initial_epoch})
    model_logger.save_state_dict(best, optimizer, lr_scheduler, filename='best.pt',
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
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        try:
            epoch_loss_csv = os.path.join(logdir, 'epoch_loss.csv')
            initial_epoch += pd.read_csv(epoch_loss_csv)['epoch'].max()
            # initial_epoch += additional_values['epoch']
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

print(str_stage, "Start training")
assert opt.epoch > 0
best_accuracy = 0.0
running_loss = 0.0
running_accuracy = 0.0

while initial_epoch <= opt.epoch:
    for phase in ['train', 'valid']:
        nbatch_loss, nbatch_acc, tlength = 0, 0, len(dataloaders[phase])
        # if initial_epoch % opt.eval_every_train != 0:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        # Loss utility
        metric_util = util_metric.MetricUtil()
        # Progress bar
        data = tqdm(dataloaders[phase], desc="Loss: ", total=len(dataloaders[phase]), ncols=80)
        for i, (inputs, labels) in enumerate(data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            batch_loss, batch_acc, running_loss, running_accuracy = metric_util.compute_metric(
                model, phase, inputs, labels, criterion, optimizer, i+1
            )
            nbatch_loss += batch_loss
            nbatch_acc += batch_acc
            lr_scheduler.step()
            # updating progress bar
            data.set_description("{} {}/{}: Loss: {:.6f}, Acc: {:.6f}".format(
                phase, initial_epoch, opt.epoch, running_loss, running_accuracy))
            # save checkpoint every 10000 batches
            # Save every 10000 cumulative batch loss into .csv file
            if i % 10000 == 9999:
                tlength -= 10000
                csv_logger.save([phase, initial_epoch, i+1, running_loss, running_accuracy,
                                 nbatch_loss / 10000, nbatch_acc / 10000])
                nbatch_loss, nbatch_acc = 0, 0
                # save most recent model as checkpoint
                checkpoint = copy.deepcopy(model.state_dict())
                model_logger.save_state_dict(checkpoint, optimizer, lr_scheduler, filename='checkpoint.pt',
                                             additional_values={'epoch': initial_epoch})
        # Save best model if exist
        if phase == 'valid' and running_accuracy > best_accuracy:
            best_accuracy = running_accuracy
            best = copy.deepcopy(model.state_dict())
            model_logger.save_state_dict(best, optimizer, lr_scheduler, filename='best.pt',
                                         additional_values={'epoch': initial_epoch})
        # Save validation ground truth and prediction
        if phase == 'valid':
            metric_logger.save_metric(metric_util.pred, metric_util.gt, initial_epoch)
        # save most recent model as checkpoint
        assert tlength > 0 & tlength < 10000
        csv_logger.save([phase, initial_epoch, -1, running_loss, running_accuracy,
                         nbatch_loss / tlength, nbatch_acc / tlength])
        checkpoint = copy.deepcopy(model.state_dict())
        model_logger.save_state_dict(checkpoint, optimizer, lr_scheduler, filename='checkpoint.pt',
                                     additional_values={'epoch': initial_epoch})
    initial_epoch += 1
print("Best validation accuracy: {:.6f}".format(best_accuracy))

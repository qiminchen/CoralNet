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
from util.util_augmentation import dataset_sampling
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

print(str_stage, "Setting up dataset data loaders")
start_time = time.time()
Dataset = datasets.get_dataset(opt.dataset)
dataset = {
    'train': Dataset(opt, mode='train'),
    'valid': Dataset(opt, mode='valid')
}

print(str_verbose, "Trained with augmented dataset: {}".format(opt.augmented))
dataset, dataloaders = dataset_sampling(dataset, opt, collate_data)

print(str_verbose, "Time spent in data IO initialization: %.2fs" %
      (time.time() - start_time))
print(str_verbose, "# training points: " + str(len(dataset['train'])))
print(str_verbose, "# training batches per epoch: " + str(len(dataloaders['train'])))
print(str_verbose, "# test batches: " + str(len(dataloaders['valid'])))

###################################################

print(str_stage, "Setting up optimizer and learning rate scheduler")
criterion = nn.CrossEntropyLoss()
if opt.optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum,
                                weight_decay=opt.wdecay)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.max_lr, epochs=opt.epoch,
                                                       steps_per_epoch=len(dataloaders['train']))
elif opt.optim == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), opt.lr*opt.lrdecay, betas=(opt.adam_beta1, opt.adam_beta2),
                                 weight_decay=opt.wdecay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrdecaystep)
else:
    raise ModuleNotFoundError(opt.optim)

###################################################

print(str_stage, "Resuming model information")
initial_epoch = 1
current_batch = 0
current_phase = 'train'
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
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if opt.optim == 'adam':
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr * opt.lrdecay
        try:
            csv_log = pd.read_csv(os.path.join(logdir, 'epoch_loss.csv'))
            initial_epoch = (csv_log['epoch'].max() + 1) if csv_log['phase'].tail(1).values[0] == 'valid' \
                else csv_log['epoch'].max()
            current_batch = 0 if csv_log['phase'].tail(1).values[0] == 'valid' \
                else csv_log['batch'].tail(1).values[0]
            current_phase = 'valid' if current_batch == len(dataloaders['train']) else 'train'
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
    # Manually adjust the learning rate if using adam
    # Double-check the learning rate
    if opt.optim == 'adam':
        for param_group in optimizer.param_groups:
            assert param_group['lr'] == opt.lr * opt.lrdecay
    # Re-sampling the dataset, test set will remain the same
    if opt.augmented:
        del dataloaders
        _, dataloaders = dataset_sampling(dataset, opt, collate_data)
    for phase in ['train', 'valid']:
        # Skip training phase one time if current phase is supposed to be valid
        if current_batch != 0 and current_phase == 'valid':
            current_batch = 0
            continue
        current_phase = phase
        batch_losses, batch_accs, dataloader_len = 0, 0, len(dataloaders[phase])
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
            # current_batch is useful only when resuming
            if current_batch+i == dataloader_len:
                break
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            batch_loss, batch_acc, running_loss, running_accuracy = metric_util.compute_metric(
                model, phase, inputs, labels, criterion, optimizer, lr_scheduler, i+1
            )
            batch_losses += batch_loss
            batch_accs += batch_acc
            # updating progress bar
            data.set_description("{} {}/{}: Loss: {:.6f}, Acc: {:.6f}".format(
                phase, initial_epoch, opt.epoch, running_loss, running_accuracy))
            # save checkpoint every 10000 batches
            # Save every 10000 cumulative batch loss into .csv file
            if (current_batch+i+1) % opt.log_batch == 0:
                csv_logger.save([phase, initial_epoch, current_batch+i+1, running_loss, running_accuracy,
                                 batch_losses / opt.log_batch, batch_accs / opt.log_batch])
                batch_losses, batch_accs = 0, 0
                # save most recent model as checkpoint
                checkpoint = copy.deepcopy(model.state_dict())
                model_logger.save_state_dict(checkpoint, optimizer, lr_scheduler,
                                             filename='checkpoint_mb'+str(i+1)+'.pt',
                                             additional_values={'epoch': initial_epoch})
            del inputs, labels
        if dataloader_len % opt.log_batch != 0:
            csv_logger.save([phase, initial_epoch, dataloader_len, running_loss, running_accuracy,
                             batch_losses / (dataloader_len % opt.log_batch),
                             batch_accs / (dataloader_len % opt.log_batch)])
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
        checkpoint = copy.deepcopy(model.state_dict())
        model_logger.save_state_dict(checkpoint, optimizer, lr_scheduler, filename='checkpoint.pt',
                                     additional_values={'epoch': initial_epoch})
        current_batch = 0
    initial_epoch += 1
print("Best validation accuracy: {:.6f}".format(best_accuracy))

import sys
import os
import time
import torch
from options import options_train
import datasets
import models
from util.util_print import str_error, str_stage, str_verbose, str_warning


###################################################

print(str_stage, "Parsing arguments")
opt, unique_opt_params = options_train.parse()
# Get all parse done, including subparsers
print(opt)

###################################################
'''
print(str_stage, "Setting device")
if opt.gpu == '-1':
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')
'''
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

print(str_stage, "Setting up loggers")

###################################################

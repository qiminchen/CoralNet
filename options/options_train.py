import argparse
import torch
from util.util_print import str_warning


def add_general_arguments(parser):
    # Parameters that will NOT be overwritten when resuming
    unique_params = {'gpu', 'resume', 'epoch', 'workers', 'batch_size', 'save_net', 'epoch_batches', 'logdir'}

    parser.add_argument('--gpu', default='0', type=str, help='gpu to use')
    parser.add_argument('--manual_seed', type=int, default=None, help='manual seed for randomness')
    parser.add_argument('--resume', type=int, default=0,
                        help='resume training by loading checkpoint.pt or best.pt. Use 0 for training from scratch, \
                              -1 for last and -2 for previous best. Use positive number for a specific epoch. \
                              Most options will be overwritten to resume training with exactly same environment')
    parser.add_argument('--suffix', default='', type=str,
                        help="Suffix for `logdir` that will be formatted with `opt`, e.g., 'lr{lr}'"
    )
    parser.add_argument('--epoch', type=int, default=0, help='number of epochs to train')

    # Dataset IO
    parser.add_argument('--dataset', type=str, default=None, help='dataset to use')
    parser.add_argument('--source', type=str, default=None,
                        help='which source to use for feature extraction or network training')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--epoch_batches', default=None, type=int, help='number of batches used per epoch')
    parser.add_argument('--eval_batches', default=None,
                        type=int, help='max number of batches used for evaluation per epoch')
    parser.add_argument('--log_time', action='store_true', help='adding time log')
    parser.add_argument('--input_size', type=int, default=224, help='image size, 224x224 or 168x168')
    parser.add_argument('--augmented', type=int, default=0, help='augmented dataset or not')

    # Network name
    parser.add_argument('--net', type=str, required=True, default='efficientnet', help='network type to use')
    parser.add_argument('--pretrained', action='store_true', help='using pretrained model or not')
    parser.add_argument('--fine_tune', action='store_true',
                        help='True -> fine tune the entire network,'
                             'False -> only fine tune the top layer')
    parser.add_argument('--nclasses', type=int, required=True, default=1000, help='number of source set class')
    parser.add_argument('--ntclasses', type=int, default=0, help='number of target set class')
    parser.add_argument('--net_path', type=str, default=None, help='fine-tune ResNet50/EfficientNet path')
    parser.add_argument('--net_version', type=str, default='b4',
                        help='for EfficientNet, value: b0, b1, b2, b3, b4, b5,'
                             'for ResNet, value: resnet18, resnet34, resnet50,'
                             'resnet101, resnet152, resnext50_32x4d, resnext101_32x8d,'
                             'wide_resnet50_2, wide_resnet101_2')
    parser.add_argument('--embed_size', type=int, default=1280, help='embedding size of EfficientNet')

    # Optimizer
    parser.add_argument('--optim', type=str, default='adam', help='optimizer to use')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD')
    parser.add_argument('--adam_beta1', type=float, default=0.5, help='beta1 of adam')
    parser.add_argument('--adam_beta2', type=float, default=0.9, help='beta2 of adam')
    parser.add_argument('--wdecay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--lrdecay', type=float, default=0.1, help='learning rate decay')
    parser.add_argument('--lrdecaystep', type=int, default=100, help='learning rate decay step size')
    parser.add_argument('--max_lr', type=float, default=1e-1, help='max learning rate for one cycle policy')

    # Logging and visualization
    parser.add_argument('--logdir', type=str, default=None,
                        help='Root directory for logging. Actual dir is [logdir]/[net_classes_dataset]/[expr_id]')
    parser.add_argument('--log_batch', type=int, default=10000, help='Log batch loss')
    parser.add_argument('--expr_id', type=int, default=0,
                        help='Experiment index. non-positive ones are overwritten by default. Use 0 for code test. ')
    parser.add_argument('--save_net', type=int, default=1, help='Period of saving network weights')
    parser.add_argument('--save_net_opt', action='store_true', help='Save optimizer state in regular network saving')
    parser.add_argument('--eval_every_train', default=10, type=int, help="Evaluate every N epochs during validation")
    parser.add_argument('--vis_every_train', default=1, type=int, help="Visualize every N epochs during training")
    parser.add_argument('--vis_batches_vali', type=int, default=10, help="# batches to visualize during validation")
    parser.add_argument('--vis_batches_train', type=int, default=10, help="# batches to visualize during training")
    parser.add_argument('--tensorboard', action='store_true',
                        help='Use tensorboard for logging. If enabled, the output log will be at '
                             '[logdir]/[tensorboard]/[net_classes_dataset]/[expr_id]')
    parser.add_argument('--vis_workers', default=4, type=int, help="# workers for the visualizer")
    parser.add_argument('--vis_param_f', default=None, type=str,
                        help="Parameter file read by the visualizer on every batch; defaults to 'visualize/config.json'")

    return parser, unique_params


def overwrite(opt, opt_f_old, unique_params):
    opt_dict = vars(opt)
    opt_dict_old = torch.load(opt_f_old)
    for k, v in opt_dict_old.items():
        if k in opt_dict:
            if (k not in unique_params) and (opt_dict[k] != v):
                print(str_warning, "Overwriting %s for resuming training: %s -> %s"
                      % (k, str(opt_dict[k]), str(v)))
                opt_dict[k] = v
        else:
            print(str_warning, "Ignoring %s, an old option that no longer exists" % k)
    opt = argparse.Namespace(**opt_dict)
    return opt


def parse():
    parser = argparse.ArgumentParser()
    parser, unique_params = add_general_arguments(parser)

    opt = parser.parse_args()
    return opt, unique_params

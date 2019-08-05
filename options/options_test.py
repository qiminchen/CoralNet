import argparse


def add_general_arguments(parser):
    # Dataset IO
    parser.add_argument('--dataset', type=str, default=None,
                        help='dataset to use')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='training batch size')

    # Network name
    parser.add_argument('--net', type=str, required=True, default='efficientnet',
                        help='network type to use')
    parser.add_argument('--pretrained', action='store_true',
                        help='using pretrained model or not')
    parser.add_argument('--fine_tune', action='store_true',
                        help='True -> fine tune the entire network,'
                             'False -> only fine tune the top layer')
    parser.add_argument('--nclasses', type=int, default=1000,
                        help='number of source set class')
    parser.add_argument('--net_path', type=str, default=None,
                        help='fine-tune ResNet50/EfficientNet path')
    parser.add_argument('--net_version', type=str, default='b4',
                        help='for EfficientNet, value: b0, b1, b2, b3, b4, b5,'
                             'for ResNet, value: resnet18, resnet34, resnet50,'
                             'resnet101, resnet152, resnext50_32x4d, resnext101_32x8d,'
                             'wide_resnet50_2, wide_resnet101_2')
    return parser


def parse():
    parser = argparse.ArgumentParser()
    parser = add_general_arguments(parser)

    opt = parser.parse_args()
    return opt

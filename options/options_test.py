import argparse


def add_general_arguments(parser):
    parser.add_argument('--gpu', default='0', type=str, help='gpu to use')
    # Dataset IO
    parser.add_argument('--dataset', type=str, default=None, help='dataset to use')
    parser.add_argument('--source', type=str, default='s16', help='which source to be extracted')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--input_size', type=int, default=224, help='image size, 224x224 or 168x168')

    # Network name
    parser.add_argument('--net', type=str, required=True, default='efficientnet', help='network type to use')
    parser.add_argument('--pretrained', action='store_true', help='using pretrained model or not')
    parser.add_argument('--fine_tune', action='store_true',
                        help='True -> fine tune the entire network,'
                             'False -> only fine tune the top layer')
    parser.add_argument('--nclasses', type=int, default=1000, help='number of source set class')
    parser.add_argument('--net_path', type=str, default=None, help='fine-tune ResNet50/EfficientNet path')
    parser.add_argument('--net_version', type=str, default='b4',
                        help='for EfficientNet, value: b0, b1, b2, b3, b4, b5,'
                             'for ResNet, value: resnet18, resnet34, resnet50,'
                             'resnet101, resnet152, resnext50_32x4d, resnext101_32x8d,'
                             'wide_resnet50_2, wide_resnet101_2')
    parser.add_argument('--embed_size', type=int, default=1280, help='embedding size of EfficientNet')
    return parser


def parse():
    parser = argparse.ArgumentParser()
    parser = add_general_arguments(parser)

    opt = parser.parse_args()
    return opt

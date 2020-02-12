import importlib


def get_model(opt):
    module = importlib.import_module('models.' + opt.net)
    if opt.net == 'resnet':
        model = module.resnet(opt.pretrained, opt.net_version,
                              opt.nclasses, opt.fine_tune)
    elif opt.net == 'efficientnet':
        model = module.efficientnet(opt.pretrained, opt.net_version,
                                    opt.nclasses, opt.fine_tune)
    elif opt.net == 'vgg':
        model = module.vgg(opt.pretrained, opt.net_version,
                           opt.nclasses, opt.fine_tune)
    else:
        raise NotImplementedError(opt.net)
    return model

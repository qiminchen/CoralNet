import importlib


def get_model(opt):
    module = importlib.import_module('models.' + opt.net)
    if opt.net == 'resnet':
        model = module.resnet(opt.pretrained, opt.net_version,
                              opt.nclasses, opt.fine_tune)
        if opt.sets == 'target':
            assert opt.ntclasses > 0
            model = module.replace_classifier(model, opt.net_path,
                                              opt.ntclasses)
    elif opt.net == 'efficientnet':
        model = module.efficientnet(opt.pretrained, opt.net_version,
                                    opt.nclasses, opt.fine_tune)
        # replace classifier in future work
    else:
        raise NotImplementedError(opt.net)
    return model

import importlib


def get_model(opt):
    module = importlib.import_module('models.' + opt.net)
    if opt.net == 'resnet50':
        model = module.resnet50(opt.pretrained, opt.nclasses)
        if opt.sets == 'target':
            assert opt.ntclasses > 0
            model = module.replace_classifier(model, opt.net_path,
                                              opt.ntclasses)
    elif opt.net == 'efficientnet':
        model = module.efficientnet(opt.pretrained, opt.efficient_version,
                                    opt.nclasses)
        # replace classifier in future work
    else:
        raise NotImplementedError()
    return model

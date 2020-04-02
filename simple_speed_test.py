import torch
import time
import numpy as np
from models.efficientnet import efficientnet
from models.vgg import vgg
import torch.nn as nn
from models.resnet import resnet


vgg16 = vgg(False, 'vgg16', 1000, False)
vgg16.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-1])
resnet50 = resnet(False, 'resnet50', 1000, False)
resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
resnet101 = resnet(False, 'resnet101', 1000, False)
resnet101 = nn.Sequential(*list(resnet101.children())[:-1])

models = {
    'efficientnetb0': efficientnet(False, 'b0', 1000, False, 1280),
    'efficientnetb1': efficientnet(False, 'b1', 1000, False, 1280),
    'efficientnetb4': efficientnet(False, 'b4', 1000, False, 1280),
    'vgg16': vgg16,
    'resnet50': resnet50,
    'resnet101': resnet101
}

batch_sizes = [1, 10, 25]
patch_sizes = [168, 224]
devices = ['cpu', 'cuda']
for key, model in models.items():
    print('==> model: {}'.format(key))
    for dev in devices:
        device = torch.device(dev)
        model = model.to(device)
        for patch_size in patch_sizes:
            for batch_size in batch_sizes:
                run_times = []
                for _ in range(10):
                    patch = torch.rand(batch_size, 3, patch_size, patch_size).to(device)
                    t0 = time.perf_counter()
                    if key.startswith('efficientnet'):
                        res = model.extract_features(patch)
                    else:
                        res = model(patch).squeeze(-1).squeeze(-1)
                    if dev == 'cuda':
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    run_times.append((t1 - t0)/batch_size)
                decimal = 4 if dev == 'cpu' else 8
                print('[device: {}, patch size: {}, batch size: {}] Time per patch: {} ± {} seconds'.format(dev,
                      patch_size, batch_size, np.round(np.mean(run_times), decimal), np.round(np.std(run_times), decimal)))

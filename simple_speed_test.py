import torch
import time
import numpy as np
from models.efficientnet import efficientnet

model = efficientnet(False, 'b0', 1000, False, 1024)

batch_sizes = [1, 10, 25]
patch_sizes = [168, 224]
for patch_size in patch_sizes:
    for batch_size in batch_sizes:
        run_times = []
        for _ in range(10):
            patch = torch.rand(batch_size, 3, patch_size, patch_size)
            t0 = time.perf_counter()
            res = model.extract_features(patch)
            # If measuring GPU run-times, add this line:
            # torch.cuda.synchronize()
            t1 = time.perf_counter()
            run_times.append((t1 - t0)/batch_size)
        print('[batch size: {}, patch size: {}] '
              'Time per patch: {:.4f} +- {:.4f} seconds'.
            format(batch_size,
                   patch_size,
                   np.mean(run_times),
                   np.std(run_times)))
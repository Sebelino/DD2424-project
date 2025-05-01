import numpy as np


class Determinism:
    def __init__(self, seed=0):
        self.seed = seed

    def sow(self):
        import os
        # Must come before any torch imports
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch
        import random
        import numpy as np
        seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Slower, but reproducible
        torch.use_deterministic_algorithms(True)
        os.environ['PYTHONHASHSEED'] = str(seed)
        return self

    @staticmethod
    def torch_generator(seed, torch_generator):
        torch_generator.manual_seed(seed)

    @staticmethod
    def data_loader_worker_init_fn(seed):
        return lambda worker_id: np.random.seed(seed + worker_id)

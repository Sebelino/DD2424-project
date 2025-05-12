from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Determinism:
    seed: int = 0

    def sow(self, seed=None):
        if seed is None:
            seed = self.seed
        import os
        # Must come before any torch imports
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch
        import random
        import numpy as np
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
    def data_loader_worker_init_fn(base_seed):
        def init_fn(worker_id):
            import random
            import torch
            seed = base_seed + worker_id
            # Python RNG
            random.seed(seed)
            # NumPy RNG
            np.random.seed(seed)
            # Torch CPU RNG
            torch.manual_seed(seed)
            # (If you ever use CUDA inside workers:)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        return init_fn

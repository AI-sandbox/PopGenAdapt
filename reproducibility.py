import os
import random

import numpy as np
import torch

def make_reproducible(seed: int = 42) -> None:
    """
    Make the results reproducible, possibly at a performance cost.

    Note that completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms.
    Furthermore, results may not be reproducible between CPU and GPU executions, even when using identical seeds.
    More details can be found at https://pytorch.org/docs/stable/notes/randomness.html.

    Parameters
    ----------
    seed : int
        random seed to use
    """

    # See https://github.com/pytorch/pytorch/issues/47672 and https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility for details. 
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

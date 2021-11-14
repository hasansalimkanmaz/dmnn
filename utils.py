import random

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    # Seeding for the reproducibility of results.
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

# File to store common utility functions

import logging
from datetime import datetime
# import argparse
import random
import numpy as np
import torch

def setup_logging(log_filename: str = 'log'):
    now = datetime.now()  # current date and time
    # log_file = 'log_' + now.strftime('%Y%m%d_%H%M%S')
    log_file = log_filename + '_' + now.strftime('%Y%m%d') + '.log'
    logging.basicConfig(filename=log_file, format='[%(levelname)s %(funcName)s:%(lineno)d] %(message)s', level=logging.INFO, filemode='w')
    return

def set_seed(seed: int = 573) -> None:
    """Sets various random seeds. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
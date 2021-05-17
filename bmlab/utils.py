import time
from functools import wraps
import logging

import numpy as np

logger = logging.getLogger(__name__)


def debug_timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug('%s needed: %f s' % (func.__name__, end - start))
        return result
    return wrapper


def array_dump(name, arr):
    """
    A utility function for dumping numpy arrays to disk.
    Used for manually creating test data in debugging mode.
    """
    with open(name + '.npy', 'wb') as f:
        np.save(f, arr)

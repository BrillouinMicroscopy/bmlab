import time
from functools import wraps

import logging

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

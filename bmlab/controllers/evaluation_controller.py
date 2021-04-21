import logging

from bmlab.session import Session

logger = logging.getLogger(__name__)


class EvaluationController(object):

    def __init__(self):
        self.session = Session.get_instance()
        return

    def evaluate(self, count=None, max_count=None):
        count.value += 1
        max_count.value += 1
        return

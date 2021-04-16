"""
Functional tests for typical headless applications.
"""

import pathlib

from bmlab import Session
from bmlab.models import Orientation

DATA_DIR = pathlib.Path(__file__).parent / 'data'


def test_typical_use_case():

    # Start session
    session = Session.get_instance()

    # Load data file
    session.set_file(DATA_DIR / 'Water.h5')

    # Select repetition
    session.set_current_repetition('0')

    # Set orientation
    session.orientation = Orientation(rotation=0, reflection={
        'vertically': True, 'horizontally': False
    })

    # Get all available calibration keys
    # TODO
    calib_keys = ['1']

    for calib_key in calib_keys:
        points = [(100, 290), (145, 255), (290, 110)]
        time = session.current_repetition().calibration.get_time(calib_key)
        for p in points:
            session.extraction_model().add_point(calib_key, time, *p)

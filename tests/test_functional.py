"""
Functional tests for typical headless applications.
"""

import pathlib

from bmlab import Session
from bmlab.models import Orientation
from bmlab.controllers import CalibrationController, ExtractionController

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

    # Controllers
    ec = ExtractionController()
    cc = CalibrationController()

    # Models
    em = session.extraction_model()
    cm = session.calibration_model()

    points = [(100, 290), (145, 255), (290, 110)]
    for calib_key in session.get_calib_keys():
        for p in points:
            ec.add_point(calib_key, p)
        ec.optimize_points(calib_key)
        ec.optimize_points(calib_key)
        ec.optimize_points(calib_key)

        assert em.get_points(calib_key) != points
        assert em.get_circle_fit(calib_key)

    # Calibration
    for calib_key in session.get_calib_keys():

        cm.add_brillouin_region(calib_key, (149, 202))
        cm.add_brillouin_region(calib_key, (250, 304))
        cm.add_rayleigh_region(calib_key, (93, 127))
        cm.add_rayleigh_region(calib_key, (351, 387))

        assert cm.get_spectra(calib_key) is None

        cc.extract_calibration_spectra(calib_key)

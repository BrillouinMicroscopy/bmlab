"""
Functional tests for typical headless applications.
"""

import pathlib

from bmlab import Session
from bmlab.geometry import Circle, discretize_arc
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

    # Extraction
    cal = session.current_repetition().calibration
    em = session.extraction_model()
    for calib_key in session.get_calib_keys():
        points = [(100, 290), (145, 255), (290, 110)]
        time = cal.get_time(calib_key)
        for p in points:
            em.add_point(calib_key, time, *p)
        imgs = cal.get_image(calib_key)
        img = imgs[0, ...]
        em.optimize_points(calib_key, img)
        em.optimize_points(calib_key, img)
        em.optimize_points(calib_key, img)

        circle_fit = em.get_circle_fit(calib_key)
        center, radius = circle_fit
        circle = Circle(center, radius)
        phis = discretize_arc(circle, img.shape, num_points=500)

        session.extraction_model().set_extraction_angles(calib_key, phis)

        assert em.get_points(calib_key) != points
        assert em.get_circle_fit(calib_key)
        assert em.get_extracted_values(calib_key) is None

        session.extract_calibration_spectrum(calib_key)

    # Calibration
    cm = session.calibration_model()
    for calib_key in session.get_calib_keys():

        cm.add_brillouin_region(calib_key, (149, 202))
        cm.add_brillouin_region(calib_key, (250, 304))
        cm.add_rayleigh_region(calib_key, (93, 127))
        cm.add_rayleigh_region(calib_key, (351, 387))

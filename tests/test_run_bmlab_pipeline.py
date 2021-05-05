import pathlib

from bmlab.controllers.calibration_controller import CalibrationController
from bmlab.controllers.evaluation_controller import EvaluationController
from bmlab.geometry import Circle, discretize_arc
from bmlab.models import Orientation
from bmlab.models.setup import AVAILABLE_SETUPS
from bmlab.session import Session


def test_run_pipeline():
    data_dir = pathlib.Path(__file__).parent / 'data'

    # Start session
    session = Session.get_instance()

    # Load data file
    session.set_file(data_dir / 'Water.h5')

    # Select repetition
    session.set_current_repetition('0')
    session.set_setup(AVAILABLE_SETUPS[0])

    # Set orientation
    session.orientation = Orientation(rotation=0, reflection={
        'vertically': True, 'horizontally': False
    })

    # Extraction
    em = session.extraction_model()
    cm = session.calibration_model()

    cal = session.current_repetition().calibration

    calibration_controller = CalibrationController()
    evaluation_controller = EvaluationController()

    for calib_key in session.get_calib_keys():
        points = [(100, 290), (162, 240), (290, 95)]
        time = cal.get_time(calib_key)
        for p in points:
            em.add_point(calib_key, time, *p)
        imgs = cal.get_image(calib_key)
        img = imgs[0, ...]
        em.optimize_points(calib_key, img)

        circle_fit = em.get_circle_fit(calib_key)
        center, radius = circle_fit
        circle = Circle(center, radius)
        phis = discretize_arc(circle, img.shape, num_points=500)
        session.extraction_model().set_extraction_angles(calib_key, phis)

        # this values should work for both repetitions
        cm.add_brillouin_region(calib_key, (190, 250))
        cm.add_brillouin_region(calib_key, (290, 350))
        cm.add_rayleigh_region(calib_key, (110, 155))
        cm.add_rayleigh_region(calib_key, (370, 410))

        calibration_controller.calibrate(calib_key)
        evaluation_controller.evaluate(abort=False)

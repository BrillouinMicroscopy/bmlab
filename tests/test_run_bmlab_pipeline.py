import pathlib

import numpy as np

from bmlab.controllers import CalibrationController, EvaluationController,\
    ExtractionController
from bmlab.models import Orientation
from bmlab.models.setup import AVAILABLE_SETUPS
from bmlab.session import Session


def run_pipeline():
    data_dir = pathlib.Path(__file__).parent / 'data'

    # Start session
    session = Session.get_instance()

    # Load data file
    session.set_file(data_dir / 'Water.h5')

    # Select repetition
    session.set_current_repetition('0')
    session.set_setup(AVAILABLE_SETUPS[0])

    # Check that we loaded the correct file
    assert session.file.date.isoformat() == '2020-11-03T15:20:30.682000+01:00'

    # Set orientation
    session.orientation = Orientation(rotation=1, reflection={
        'vertically': False, 'horizontally': False
    })

    # Models
    em = session.extraction_model()
    cm = session.calibration_model()
    pm = session.peak_selection_model()

    ec = ExtractionController()
    cc = CalibrationController()
    evc = EvaluationController()

    points = {
        '1': [
            (31, 358),
            (107, 293),
            (165, 237),
            (254, 137),
            (291, 92),
            (323, 51),
        ],
        '2': [
            (30, 358),
            (107, 293),
            (165, 237),
            (255, 137),
            (291, 92),
            (323, 51),
            (335, 35),
        ]
    }

    # First add all extraction points because this
    # can influence the extraction for other calibrations
    for calib_key in session.get_calib_keys():
        ec.find_points(calib_key)

        p = em.get_points(calib_key)

        assert p == points[calib_key]

    brillouin_region_centers = [230, 330]
    rayleigh_region_centers = [130, 390]

    # Then do the calibration
    for calib_key in session.get_calib_keys():
        cc.find_peaks(calib_key)

        brillouin_regions = cm.get_brillouin_regions(calib_key)
        rayleigh_regions = cm.get_rayleigh_regions(calib_key)

        for center in brillouin_region_centers:
            assert region_found(center, brillouin_regions)

        for center in rayleigh_region_centers:
            assert region_found(center, rayleigh_regions)

        cc.calibrate(calib_key)

    pm.add_brillouin_region((190, 250))
    pm.add_brillouin_region((290, 350))
    pm.add_rayleigh_region((110, 155))
    pm.add_rayleigh_region((370, 410))

    evc.evaluate()
    return session


def region_found(center, regions):
    for region in regions:
        if region[0] <= center <= region[1]:
            return True
    return False


def test_run_pipeline():

    session = run_pipeline()
    evm = session.evaluation_model()
    np.testing.assert_allclose(
        evm.results['brillouin_shift_f'], 5.03e9, atol=50E6)

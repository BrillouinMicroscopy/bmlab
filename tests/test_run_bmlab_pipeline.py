import pathlib

import numpy as np

from bmlab.controllers import CalibrationController, EvaluationController,\
    ExtractionController
from bmlab.models import Orientation
from bmlab.models.setup import AVAILABLE_SETUPS
from bmlab.session import Session


def run_pipeline():
    # Start session
    session = Session.get_instance()

    # Load data file
    session.set_file(pathlib.Path(__file__).parent / 'data' / 'Water.h5')

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
    pm = session.peak_selection_model()

    ec = ExtractionController()
    cc = CalibrationController()
    evc = EvaluationController()

    # First add all extraction points because this
    # can influence the extraction for other calibrations
    for calib_key in session.get_calib_keys():
        ec.find_points(calib_key)

    # Then do the calibration
    for calib_key in session.get_calib_keys():
        cc.find_peaks(calib_key)

        cc.calibrate(calib_key)

    pm.add_brillouin_region((190, 250))
    pm.add_brillouin_region((290, 350))
    pm.add_rayleigh_region((110, 155))
    pm.add_rayleigh_region((370, 410))

    evc.evaluate()
    return session


def test_run_pipeline():

    session = run_pipeline()
    evm = session.evaluation_model()
    np.testing.assert_allclose(
        evm.results['brillouin_shift_f'], 5.03e9, atol=50E6)

import pathlib

from bmlab.session import Session
from bmlab.controllers import ExtractionController, CalibrationController
from bmlab.models import Orientation
from bmlab.models.setup import AVAILABLE_SETUPS


def test_find_peaks_real_data():
    # Start session
    session = Session.get_instance()

    # Load data file
    session.set_file(pathlib.Path(__file__).parent / 'data' / 'Water.h5')

    # Select repetition
    session.set_current_repetition('0')
    session.set_setup(AVAILABLE_SETUPS[0])

    # Set orientation
    session.orientation = Orientation(rotation=1, reflection={
        'vertically': False, 'horizontally': False
    })

    cm = session.calibration_model()

    ec = ExtractionController()
    cc = CalibrationController()

    # First add all extraction points because this
    # can influence the extraction for other calibrations
    for calib_key in session.get_calib_keys():
        ec.find_points(calib_key)

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


def region_found(center, regions):
    for region in regions:
        if region[0] <= center <= region[1]:
            return True
    return False


def test_calibrate():
    calib_key = '0'
    cc = CalibrationController()
    cc.calibrate(calib_key)

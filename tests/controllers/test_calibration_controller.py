from bmlab.controllers.calibration_controller import CalibrationController


def test_calibrate():

    calib_key = '0'
    calibration_controller = CalibrationController()
    calibration_controller.calibrate(calib_key)

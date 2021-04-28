from bmlab.models.calibration_model import CalibrationModel


def test_calibration_model_add_brillouin_region():
    cm = CalibrationModel()
    cm.add_brillouin_region(0, [0.9, 3])
    cm.add_brillouin_region(0, (2, 5))

    brillouin_region_0 = cm.get_brillouin_regions(0)

    assert brillouin_region_0 == [(1, 5)]

    cm.add_brillouin_region(0, (7, 9))

    assert brillouin_region_0 == [(1, 5), (7, 9)]

    cm.add_brillouin_region(0, (2, 3))

    assert brillouin_region_0 == [(1, 5), (7, 9)]


def test_calibration_model_set_brillouin_region():
    cm = CalibrationModel()
    cm.add_brillouin_region(0, [0.9, 3])

    brillouin_region_0 = cm.get_brillouin_regions(0)

    assert brillouin_region_0 == [(1, 3)]


def test_calibration_model_add_rayleigh_region():
    cm = CalibrationModel()
    cm.add_rayleigh_region(0, [0.9, 3])
    cm.add_rayleigh_region(0, (2, 5))

    rayleigh_region_0 = cm.get_rayleigh_regions(0)

    assert rayleigh_region_0 == [(1, 5)]

    cm.add_rayleigh_region(0, (7, 9))

    assert rayleigh_region_0 == [(1, 5), (7, 9)]

    cm.add_rayleigh_region(0, (2, 3))

    assert rayleigh_region_0 == [(1, 5), (7, 9)]


def test_calibration_model_set_rayleigh_region():
    cm = CalibrationModel()
    cm.add_rayleigh_region(0, [0.9, 3])

    rayleigh_region_0 = cm.get_rayleigh_regions(0)

    assert rayleigh_region_0 == [(1, 3)]

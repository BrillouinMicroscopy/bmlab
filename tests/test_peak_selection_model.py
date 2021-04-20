from bmlab.models.peak_selection_model import PeakSelectionModel


def test_calibration_model_add_brillouin_region():
    pm = PeakSelectionModel()
    pm.add_brillouin_region([0.9, 3])
    pm.add_brillouin_region((2, 5))

    brillouin_region_0 = pm.get_brillouin_regions()

    assert brillouin_region_0 == [(1, 5)]

    pm.add_brillouin_region((7, 9))

    assert brillouin_region_0 == [(1, 5), (7, 9)]

    pm.add_brillouin_region((2, 3))

    assert brillouin_region_0 == [(1, 5), (7, 9)]


def test_calibration_model_set_brillouin_region():
    pm = PeakSelectionModel()
    pm.add_brillouin_region([0.9, 3])

    brillouin_region_0 = pm.get_brillouin_regions()

    assert brillouin_region_0 == [(1, 3)]


def test_calibration_model_add_rayleigh_region():
    pm = PeakSelectionModel()
    pm.add_rayleigh_region([0.9, 3])
    pm.add_rayleigh_region((2, 5))

    rayleigh_region_0 = pm.get_rayleigh_regions()

    assert rayleigh_region_0 == [(1, 5)]

    pm.add_rayleigh_region((7, 9))

    assert rayleigh_region_0 == [(1, 5), (7, 9)]

    pm.add_rayleigh_region((2, 3))

    assert rayleigh_region_0 == [(1, 5), (7, 9)]


def test_calibration_model_set_rayleigh_region():
    pm = PeakSelectionModel()
    pm.add_rayleigh_region([0.9, 3])

    rayleigh_region_0 = pm.get_rayleigh_regions()

    assert rayleigh_region_0 == [(1, 3)]

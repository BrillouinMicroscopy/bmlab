from bmlab.models.peak_selection_model import PeakSelectionModel


def test_peak_selection_model_add_brillouin_region():
    pm = PeakSelectionModel()
    pm.add_brillouin_region([4.9, 7])
    pm.add_brillouin_region((6, 9))

    brillouin_region_0 = pm.get_brillouin_regions()

    assert brillouin_region_0 == [(5, 9)]

    pm.add_brillouin_region((1, 3))

    assert brillouin_region_0 == [(1, 3), (5, 9)]

    pm.add_brillouin_region((2, 3))

    assert brillouin_region_0 == [(1, 3), (5, 9)]


def test_peak_selection_model_set_brillouin_region():
    pm = PeakSelectionModel()
    pm.add_brillouin_region((6, 9))
    pm.set_brillouin_region(0, [0.9, 3])
    pm.set_brillouin_region(1, [4, 5])

    brillouin_region_0 = pm.get_brillouin_regions()

    assert brillouin_region_0 == [(1, 3), (4, 5)]

    pm.set_brillouin_region(0, [7, 9])

    assert brillouin_region_0 == [(4, 5), (7, 9)]


def test_peak_selection_model_add_rayleigh_region():
    pm = PeakSelectionModel()
    pm.add_rayleigh_region([4.9, 7])
    pm.add_rayleigh_region((6, 9))

    rayleigh_region_0 = pm.get_rayleigh_regions()

    assert rayleigh_region_0 == [(5, 9)]

    pm.add_rayleigh_region((1, 3))

    assert rayleigh_region_0 == [(1, 3), (5, 9)]

    pm.add_rayleigh_region((2, 3))

    assert rayleigh_region_0 == [(1, 3), (5, 9)]


def test_peak_selection_model_set_rayleigh_region():
    pm = PeakSelectionModel()
    pm.add_rayleigh_region((6, 9))
    pm.set_rayleigh_region(0, [0.9, 3])
    pm.set_rayleigh_region(1, [4, 5])

    rayleigh_region_0 = pm.get_rayleigh_regions()

    assert rayleigh_region_0 == [(1, 3), (4, 5)]

    pm.set_rayleigh_region(0, [7, 9])

    assert rayleigh_region_0 == [(4, 5), (7, 9)]

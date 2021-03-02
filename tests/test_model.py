import numpy as np

from bmlab.model import ExtractionModel


def test_extraction_model_triggers_circle_fit():

    em = ExtractionModel()
    em.add_point('0', 0, 1)
    em.add_point('0', 1, 0)

    assert (0, 1) in em.get_points('0')
    assert (1, 0) in em.get_points('0')
    assert em.get_circle_fit('0') is None

    em.add_point('0', 0, -1)

    assert (0, -1) in em.get_points('0')
    cf = em.get_circle_fit('0')
    center, radius = cf

    np.testing.assert_allclose(center, (0, 0), atol=1.E-3)
    np.testing.assert_almost_equal(radius, 1, decimal=4)


def test_optimize_points_in_extraction_model():

    img = np.zeros((100, 100), dtype=np.int)
    img[20, 20] = 1
    img[80, 80] = 1

    em = ExtractionModel()
    em.add_point('0', 15, 15)
    em.add_point('0', 75, 85)

    em.optimize_points('0', img, radius=10)
    opt_points = em.get_points('0')

    assert (20, 20) == opt_points[0]
    assert (80, 80) == opt_points[1]


def test_clear_points_from_extraction_model():
    em = ExtractionModel()
    em.add_point('0', 15, 15)
    em.add_point('0', 75, 85)

    em.clear_points('0')
    assert em.get_points('0') == []

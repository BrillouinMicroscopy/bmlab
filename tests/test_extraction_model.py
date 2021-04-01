import numpy as np

from bmlab.models.extraction_model import ExtractionModel
from bmlab.geometry import Circle, discretize_arc

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

    img = np.zeros((100, 100), dtype=int)
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


def test_set_arc_width():
    em = ExtractionModel()

    # Test default width
    assert em.arc_width == 3

    em.set_arc_width(2)

    assert em.arc_width == 2


def test_get_arc_by_calib_key():
    calib_key = '0'
    image_shape = (20,20)
    tolerance = 0.0001

    em = ExtractionModel()

    circle = Circle((0, 0), 10)

    phis = [0, np.pi/4, np.pi/2]
    for phi in phis:
        (xdata, ydata) = circle.point(phi)
        em.add_point('0', xdata, ydata)

    # Create angles at which to calculate the arc
    phis = discretize_arc(circle, image_shape, num_points=2)
    em.set_extraction_angles(calib_key, phis)

    # Get the arc
    em.set_arc_width(2)
    arc = em.get_arc_by_calib_key(calib_key)

    expected_arc = [
        np.array([[8.0, 0.0], [9.0, 0.0], [10.0, 0.0], [11.0, 0.0], [12.0, 0.0]]),
        np.array([[0.0, 8.0], [0.0, 9.0], [0.0, 10.0], [0.0, 11.0], [0.0, 12.0]])
    ]

    assert len(arc) == len(expected_arc)

    # Check that every line matches expected result
    for i in range(len(arc)):
        assert (abs(arc[i] - expected_arc[i]) < tolerance).all()
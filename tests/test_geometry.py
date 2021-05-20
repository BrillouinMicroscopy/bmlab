import numpy as np
from bmlab.geometry import Circle, Rectangle, discretize_arc


def test_intersect_circle_rectangle():
    circle = Circle((-1, -1), 2)
    rect = Rectangle((1, 1))
    actual = circle.intersection(rect)

    cut = 0.72955
    expected = [(0, cut), (cut, 0)]
    np.testing.assert_array_almost_equal(actual, expected, decimal=5)


def test_circle_angle():
    circle = Circle((0, 0), 1)
    sq2 = np.sqrt(2.)
    actual = circle.angle((sq2, sq2))
    np.testing.assert_almost_equal(actual, np.pi / 4.)

    actual = circle.angle((0, 1))
    np.testing.assert_almost_equal(actual, np.pi / 2.)

    actual = circle.angle((0, -1))
    np.testing.assert_almost_equal(actual, 3 * np.pi / 2.)


def test_circle_point():
    circle = Circle((0, 0), 1)
    actual = circle.point(np.pi / 4.)
    sq2 = np.sqrt(2.)
    np.testing.assert_array_almost_equal(actual, (sq2/2., sq2/2.))

    actual = circle.point(0, True)
    np.testing.assert_array_equal(actual, (1, 0))


def test_discretize_arc():
    circle = Circle((0, 0), 100)
    img_shape = (100, 100)
    num_points = 100
    expected = np.linspace(np.pi/2., 0, num_points)
    actual = discretize_arc(circle, img_shape, num_points)
    np.testing.assert_array_equal(actual, expected)

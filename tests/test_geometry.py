import numpy as np
from bmlab.geometry import Circle, Rectangle, discretize_arc


def test_circle():
    assert Circle((1, 1), 1).is_valid()
    assert not Circle((1, 1), 0).is_valid()
    assert not Circle((1, 1), np.nan).is_valid()
    assert not Circle((1, np.inf), 1).is_valid()
    assert not Circle((-np.inf, 0), 1).is_valid()


def test_intersect_circle_rectangle():
    circle = Circle((-1, -1), 2)
    rect = Rectangle((1, 1))
    actual = circle.intersection(rect)

    cut = 0.72955
    expected = [(0, cut), (cut, 0)]
    np.testing.assert_array_almost_equal(actual, expected, decimal=5)


def test_circle_angle():
    circle = Circle((0, 0), 1)

    actual = circle.angle((1, 0))
    np.testing.assert_almost_equal(actual, 0.0)

    actual = circle.angle((1, 1))
    np.testing.assert_almost_equal(actual, np.pi / 4.)

    actual = circle.angle((0, 1))
    np.testing.assert_almost_equal(actual, np.pi / 2.)

    actual = circle.angle((-1, 0))
    np.testing.assert_almost_equal(actual, np.pi)

    actual = circle.angle((0, -1))
    np.testing.assert_almost_equal(actual, 3 * np.pi / 2.)

    actual = circle.angle((1, -1))
    np.testing.assert_almost_equal(actual, 3 * np.pi / 2. + np.pi/4)

    circle = Circle((1, 0), 1)
    actual = circle.angle((1, 1))
    np.testing.assert_almost_equal(actual, np.pi / 2)


def test_circle_er():
    circle = Circle((0, 0), 1)

    np.testing.assert_allclose(circle.e_r(0), [1, 0])
    np.testing.assert_allclose(circle.e_r(np.pi / 2), [0, 1], atol=1e-12)


def test_circle_point():
    circle = Circle((0, 0), 1)
    actual, _ = circle.point(np.pi / 4.)
    sq2 = np.sqrt(2.)
    np.testing.assert_array_almost_equal(actual, (sq2/2., sq2/2.))

    actual = circle.point(0, True)
    np.testing.assert_array_equal(actual, (1, 0))


def test_discretize_arc():
    circle = Circle((0, 0), 50)
    img_shape = (100, 100)
    num_points = 100
    expected = np.linspace(np.pi/2., 0, num_points)
    actual = discretize_arc(circle, img_shape, num_points)
    np.testing.assert_array_equal(actual, expected)

    circle = Circle((0, 0), 70)
    img_shape = (100, 100)
    num_points = 2
    expected = np.linspace(np.pi/2., 0, num_points)
    actual = discretize_arc(circle, img_shape, num_points)
    np.testing.assert_array_equal(actual, expected)

    r = 70
    shift = (r**2 / 2)**0.5
    circle = Circle((shift, 0), r)
    img_shape = (200, 200)
    num_points = 2
    expected = np.linspace(np.pi / 2. + np.pi / 4., 0, num_points)
    actual = discretize_arc(circle, img_shape, num_points)
    np.testing.assert_array_equal(actual, expected)

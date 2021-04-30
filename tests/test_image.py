import pathlib

import numpy as np
import pytest

from bmlab.image import set_orientation, find_max_in_radius
from bmlab.image import autofind_orientation
from bmlab.image import extract_lines_along_arc
from bmlab.file import BrillouinFile
from bmlab.models.orientation import Orientation
from bmlab.geometry import Circle


def data_file_path(file_name):
    return pathlib.Path(__file__).parent / 'data' / file_name


def test_set_orientation_valid_argument():
    test_image = np.zeros([2, 3, 3])
    with pytest.raises(ValueError):
        set_orientation(test_image)


def test_set_orientation():
    test_image = np.zeros([2, 3])
    test_image[0, 0] = 1

    test_image_a = set_orientation(test_image, 1)
    test_image_b = set_orientation(test_image, 0, True, True)

    np.testing.assert_array_equal(test_image_a, np.array([[0., 1.],
                                                          [0., 0.],
                                                          [0., 0.]]))

    np.testing.assert_array_equal(test_image_b, np.array([[0., 0., 0.],
                                                          [0., 0., 1.]]))


def test_find_max_in_radius():
    img = np.zeros((100, 100))
    img[20, 30] = 1
    xy0 = 25, 35
    actual = find_max_in_radius(img, xy0, 15)
    expected = 20, 30
    assert actual == expected

    actual = find_max_in_radius(img, expected, 15)
    assert actual == expected


def test_autofind_orientation():

    bf = BrillouinFile(data_file_path('Water.h5'))
    imgs = bf.get_repetition('0').calibration.get_image('1')
    img = imgs[0, ...]

    actual = autofind_orientation(img)

    expected = Orientation()
    expected.set_reflection(vertically=True)

    assert actual.rotation == expected.rotation
    assert actual.reflection == expected.reflection


def test_extract_lines_along_arc():

    # Arrange
    orient = Orientation(reflection={'vertically': False,
                                     'horizontally': False})
    circle = Circle((0, 0), 100)
    x = y = np.arange(110, dtype=int)
    X, Y = np.meshgrid(x, y, indexing='ij')
    phis = np.linspace(0, np.pi/2., 30)
    img = np.sqrt(X**2 + Y**2)
    img_2 = X**2 + Y**2

    arc = []
    num_points = 3
    for phi in phis:
        e_r = circle.e_r(phi)
        mid_point = circle.point(phi)
        points = [mid_point + e_r *
                  k for k in np.arange(-num_points, num_points + 1)]
        arc.append(np.array(points))
    arc = np.array(arc)

    # Act
    actual = extract_lines_along_arc(img, orient, arc)
    actual_2 = extract_lines_along_arc(img_2, orient, arc)

    # Assert
    one = np.ones_like(phis)
    np.testing.assert_allclose(actual, 100*one, rtol=1.E-4)
    np.testing.assert_allclose(actual_2, 10**4 * one, rtol=1.E-3)

    # Quadratic scaling should shift the sum to the outside
    assert np.all(actual_2 > actual)

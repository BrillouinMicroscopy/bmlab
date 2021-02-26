import pathlib

import numpy as np
import pytest

from bmlab.image import set_orientation, find_max_in_radius
from bmlab.image import autofind_orientation
from bmlab.file import BrillouinFile
from bmlab.model import Orientation


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

import numpy as np
import pytest

from bmlab.image import set_orientation, find_max_in_radius, fit_circle


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


def test_circle_fit():
    expect_r = 550
    expect_c = (-180, -220)
    n_test_data_points = 6
    noise_strength = 10

    x_noise = np.random.random(n_test_data_points) * noise_strength
    y_noise = np.random.random(n_test_data_points) * noise_strength
    test_points = [(expect_r * np.cos(phi) + x_noise[i] + expect_c[0],
                    expect_r * np.sin(phi) + y_noise[i] + expect_c[1])
                   for i, phi in enumerate(
            np.linspace(0, np.pi / 2, n_test_data_points))
                   ]
    actual_c, actual_r = fit_circle(test_points)

    np.testing.assert_allclose(actual_c, expect_c, rtol=0.05)

    np.testing.assert_allclose(actual_r, expect_r, rtol=0.05)

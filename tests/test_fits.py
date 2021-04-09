import numpy as np
# import matplotlib.pyplot as plt

from bmlab.fits import lorentz, fit_lorentz, fit_circle,\
    fit_double_lorentz, calculate_exact_circle


def test_fit_lorentz():
    # Arrange
    x = np.linspace(0, 30, 100)
    w0 = 15.
    fwhm = 4
    offset = 10.
    intensity = 10.
    y_data = lorentz(x, w0, fwhm, intensity) + offset

    # plt.plot(x, y_data)
    # plt.show()

    actual_w0, actual_fwhm, actual_intensity, actual_offset =\
        fit_lorentz(x, y_data)
    np.testing.assert_almost_equal(actual_w0, w0, decimal=3)
    np.testing.assert_almost_equal(actual_fwhm, fwhm, decimal=3)
    np.testing.assert_almost_equal(actual_intensity, intensity, decimal=3)
    np.testing.assert_almost_equal(actual_offset, offset, decimal=3)


def test_fit_double_lorentz():
    # Arrange
    x = np.linspace(0, 30, 100)
    w0_left = 10.
    intensity_left = 8.
    fwhm_left = 2.
    w0_right = 20.
    intensity_right = 12.
    fwhm_right = 4.
    offset = 10.
    y_data = lorentz(x, w0_left, fwhm_left, intensity_left)
    y_data += lorentz(x, w0_right, fwhm_right, intensity_right) + offset

    # plt.plot(w, y_data)
    # plt.show()

    fit_left, fit_right, actual_offset\
        = fit_double_lorentz(x, y_data)

    np.testing.assert_almost_equal(fit_left[0], w0_left, decimal=3)
    np.testing.assert_almost_equal(fit_left[1], fwhm_left, decimal=3)
    np.testing.assert_almost_equal(fit_left[2], intensity_left, decimal=3)

    np.testing.assert_almost_equal(fit_right[0], w0_right, decimal=3)
    np.testing.assert_almost_equal(fit_right[1], fwhm_right, decimal=3)
    np.testing.assert_almost_equal(fit_right[2], intensity_right, decimal=3)

    np.testing.assert_almost_equal(actual_offset, offset, decimal=3)


def test_circle_fit():
    expect_r = 550
    expect_c = (-200, -220)
    n_test_data_points = 6
    noise_strength = 15
    np.random.seed(1)
    x_noise = np.random.random(n_test_data_points) * noise_strength
    y_noise = np.random.random(n_test_data_points) * noise_strength

    test_points_noise = [(expect_r * np.cos(phi) + x_noise[i] + expect_c[0],
                          expect_r * np.sin(phi) + y_noise[i] + expect_c[1])
                         for i, phi in enumerate(
        np.linspace(0.1, np.pi / 2, n_test_data_points))
    ]

    actual_c_noise, actual_r_noise = fit_circle(test_points_noise)

    np.testing.assert_allclose(actual_c_noise, expect_c, rtol=0.05)
    np.testing.assert_allclose(actual_r_noise, expect_r, rtol=0.05)

    test_points = [(expect_r * np.cos(phi) + 0 * x_noise[i] + expect_c[0],
                    expect_r * np.sin(phi) + 0 * y_noise[i] + expect_c[1])
                   for i, phi in enumerate(
        np.linspace(0.1, np.pi / 2, int(n_test_data_points / 2) + 1))
    ]

    actual_c, actual_r = fit_circle(test_points)
    np.testing.assert_allclose(actual_c, expect_c, rtol=1e-2)
    np.testing.assert_allclose(actual_r, expect_r, rtol=1e-2)

    # Test that radius is always positive
    test_points_2 = [(np.sqrt(2), 0), (1, 1), (0, np.sqrt(2))]
    actual_c_2, actual_r_2 = fit_circle(test_points_2)

    np.testing.assert_allclose(actual_c_2, (0, 0), rtol=0.001, atol=0.0001)
    assert(actual_r_2 > 0)
    np.testing.assert_allclose(actual_r_2, np.sqrt(2),
                               rtol=0.0001, atol=0.000001)

    test_points_3 = [
        (10.0, 0.0),
        (7.0710678118654755, 7.0710678118654755),
        (6.123233995736766e-16, 10.0)
    ]
    actual_c_3, actual_r_3 = fit_circle(test_points_3)

    np.testing.assert_allclose(actual_c_3, (0, 0), rtol=0.001, atol=0.0001)
    assert(actual_r_3 > 0)
    np.testing.assert_allclose(actual_r_3, 10, rtol=0.0001, atol=0.000001)


def test_calculate_exact_circle():
    points = [[1, 0], [0, 1], [-1, 0]]
    center, radius = calculate_exact_circle(points)

    np.testing.assert_allclose(center, [0, 0], rtol=1e-2)
    np.testing.assert_allclose(radius, 1, rtol=1e-2)

    points = [[-1, -2], [-2, -1], [-3, -2]]
    center, radius = calculate_exact_circle(points)

    np.testing.assert_allclose(center, [-2, -2], rtol=1e-2)
    np.testing.assert_allclose(radius, 1, rtol=1e-2)

    points = [[8, -2], [-2, 8], [-12, -2]]
    center, radius = calculate_exact_circle(points)

    np.testing.assert_allclose(center, [-2, -2], rtol=1e-2)
    np.testing.assert_allclose(radius, 10, rtol=1e-2)

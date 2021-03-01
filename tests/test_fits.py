import numpy as np
# import matplotlib.pyplot as plt

from bmlab.fits import lorentz, fit_lorentz, fit_circle, fit_double_lorentz


def test_fit_lorentz():
    # Arrange
    w = np.linspace(10, 20, 100)
    w_0 = 14.
    gam = 0.2
    offset = 10.
    y_data = lorentz(w, w_0, gam, offset)

    # plt.plot(w, y_data)
    # plt.show()

    actual_w0, actual_gam, actual_offset = fit_lorentz(w, y_data)
    np.testing.assert_almost_equal(actual_w0, w_0, decimal=3)
    np.testing.assert_almost_equal(actual_gam, gam, decimal=3)
    np.testing.assert_almost_equal(actual_offset, offset, decimal=3)


def test_fit_double_lorentz():
    # Arrange
    w = np.linspace(10, 20, 100)
    w0_left = 14.
    w0_right = 16.
    gam = 0.5
    offset = 10.
    y_data = lorentz(w, w0_left, gam, offset)
    y_data += lorentz(w, w0_right, gam, 0)

    # plt.plot(w, y_data)
    # plt.show()

    fit_left, fit_right = fit_double_lorentz(w, y_data)

    np.testing.assert_almost_equal(fit_left[0], w0_left, decimal=3)
    np.testing.assert_almost_equal(fit_left[1], gam, decimal=3)
    np.testing.assert_almost_equal(fit_left[2], offset, decimal=3)

    np.testing.assert_almost_equal(fit_right[0], w0_right, decimal=3)
    np.testing.assert_almost_equal(fit_right[1], gam, decimal=3)
    np.testing.assert_almost_equal(fit_right[2], offset, decimal=3)


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

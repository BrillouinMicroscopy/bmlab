import pathlib

import numpy as np

from bmlab.fits import lorentz, fit_lorentz, fit_circle,\
    fit_double_lorentz, calculate_exact_circle, fit_vipa, VIPA,\
    are_points_on_line

from bmlab.models.setup import AVAILABLE_SETUPS

import pytest


def test_fit_lorentz():
    # Arrange
    x = np.linspace(0, 30, 100)
    w0 = 15.
    fwhm = 4
    offset = 10.
    intensity = 10.
    y_data = lorentz(x, w0, fwhm, intensity) + offset

    actual_w0, actual_fwhm, actual_intensity, actual_offset =\
        fit_lorentz(x, y_data)
    np.testing.assert_almost_equal(actual_w0, w0, decimal=3)
    np.testing.assert_almost_equal(actual_fwhm, fwhm, decimal=3)
    np.testing.assert_almost_equal(actual_intensity, intensity, decimal=3)
    np.testing.assert_almost_equal(actual_offset, offset, decimal=3)


def test_fit_lorentz_real_image_data():
    """ The data for this test case has been extracted manually from the running
        BMicro application.
    """

    data_dir = pathlib.Path(__file__).parent / 'data'

    region = np.load(data_dir / 'rayleigh_reg0_region.npy')
    xdata = np.load(data_dir / 'rayleigh_reg0_xdata.npy')
    ydata = np.load(data_dir / 'rayleigh_reg0_ydata.npy')

    w0, fwhm, intensity, offset = fit_lorentz(xdata[range(*region)],
                                              ydata[range(*region)])

    assert w0 == pytest.approx(115, 0.2)
    assert fwhm == pytest.approx(6, 0.5)
    assert intensity == pytest.approx(1186, 1)
    assert offset == pytest.approx(47, 1)


def test_fit_double_lorentz():
    # Arrange
    x = np.linspace(0, 60, 200)
    w0_left = 10.
    intensity_left = 8.
    fwhm_left = 2.
    w0_right = 20.
    intensity_right = 12.
    fwhm_right = 4.
    offset = 10.
    y_data = lorentz(x, w0_left, fwhm_left, intensity_left)
    y_data += lorentz(x, w0_right, fwhm_right, intensity_right) + offset

    w0s, fwhms, intens, actual_offset\
        = fit_double_lorentz(x, y_data)

    np.testing.assert_almost_equal(w0s[0], w0_left, decimal=3)
    np.testing.assert_almost_equal(fwhms[0], fwhm_left, decimal=3)
    np.testing.assert_almost_equal(intens[0], intensity_left, decimal=3)

    np.testing.assert_almost_equal(w0s[1], w0_right, decimal=3)
    np.testing.assert_almost_equal(fwhms[1], fwhm_right, decimal=3)
    np.testing.assert_almost_equal(intens[1], intensity_right, decimal=3)

    np.testing.assert_almost_equal(actual_offset, offset, decimal=3)


def test_fit_double_lorentz_with_bounds():
    # Arrange
    x = np.linspace(0, 60, 500)
    w0_left = 20.
    intensity_left = 8.
    fwhm_left = 4.0
    w0_right = 40.
    intensity_right = 12.
    fwhm_right = 5.0
    offset = 10.
    y_data = lorentz(x, w0_left, fwhm_left, intensity_left)
    y_data += lorentz(x, w0_right, fwhm_right, intensity_right) + offset

    bounds = ((19, 19.9), (-np.Inf, np.Inf))

    w0s, fwhms, intens, actual_offset \
        = fit_double_lorentz(x, y_data, bounds)

    np.testing.assert_almost_equal(w0s[0], 19.9, decimal=2)
    np.testing.assert_almost_equal(fwhms[0], fwhm_left, decimal=1)
    np.testing.assert_almost_equal(intens[0], intensity_left, decimal=1)

    np.testing.assert_almost_equal(w0s[1], w0_right, decimal=1)
    np.testing.assert_almost_equal(fwhms[1], fwhm_right, decimal=1)
    np.testing.assert_almost_equal(intens[1], intensity_right, decimal=1)

    np.testing.assert_almost_equal(actual_offset, offset, decimal=1)

    bounds = ((-np.Inf, np.Inf), (20.1, 21))

    w0s, fwhms, intens, actual_offset \
        = fit_double_lorentz(x, y_data, bounds)

    np.testing.assert_almost_equal(w0s[0], w0_right, decimal=1)
    np.testing.assert_almost_equal(fwhms[0], fwhm_right, decimal=1)
    np.testing.assert_almost_equal(intens[0], intensity_right, decimal=1)

    np.testing.assert_almost_equal(w0s[1], 20.1, decimal=1)
    np.testing.assert_almost_equal(fwhms[1], fwhm_left, decimal=1)
    np.testing.assert_almost_equal(intens[1], intensity_left, decimal=1)

    np.testing.assert_almost_equal(actual_offset, offset, decimal=1)


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


def test_are_points_on_line():
    assert are_points_on_line([(0, 0), (1, 1), (2, 2)])
    assert are_points_on_line([(0, 0), (1, 1)])
    assert not are_points_on_line([(0, 0), (1, 1), (3, 2)])
    assert not are_points_on_line([(0, 0), (1, 1), (3, 2), (4, 4)])
    assert are_points_on_line([(0, 0), (1, 1), (2, 2), (3, 4)])


def test_fit_vipa():
    setup = AVAILABLE_SETUPS[0]
    peaks = np.array([
        84.2957567375179,
        147.651886970066,
        166.035559534916,
        229.678232333124,
        244.118149639528,
        287.316083765756
    ])

    vipa_params = fit_vipa(peaks, setup)

    # Values extracted from the previous Matlab version
    vipa_params_expected =\
        np.array([
            2.602626743299098e-15, -2.565391389072813e-22,
            -6.473779072623057e-25, 14.89190812511701e+9
        ])

    actual = VIPA(peaks, vipa_params) - setup.f0
    expected = VIPA(peaks, vipa_params_expected) - setup.f0

    # Values are in GHz, differences below 5 MHz should be good
    np.testing.assert_allclose(actual, expected, atol=5e3)

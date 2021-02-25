import numpy as np
# import matplotlib.pyplot as plt

from bmlab.fits import lorentz, fit_lorentz


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

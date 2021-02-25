import numpy as np


from bmlab.fits import lorentz, fit_lorentz

def test_fit_lorentz():
    # Arrange
    w = np.linspace(10, 20, 30)
    w_0 = 14.
    gam = 2.
    y_data = lorentz(w, w_0, gam)

    actual_w0, actual_gam = fit_lorentz(w, y_data)
    np.testing.assert_almost_equal(actual_w0, w_0)
    np.testing.assert_almost_equal(actual_gam, gam)

import numpy as np
from scipy.optimize import least_squares


class FitError(Exception):
    pass


def lorentz(w, w_0, gam, offset):
    return 1. / ((w**2 - w_0**2)**2 + gam**2 * w_0**2) + offset


def fit_lorentz(w, y):
    w_0_guess = w[np.argmax(y)]
    offset_guess = (y[0] + y[-1]) / 2.
    gam_guess = (w_0_guess**2 * ((np.max(y) - offset_guess))) ** -0.5

    def error(x, w, y): return np.sum((y - lorentz(w, *x))**2)

    opt_result = least_squares(error, x0=(
        w_0_guess, gam_guess, offset_guess),
        args=(w, y))

    if not opt_result.success:
        raise FitError('Lorentz fit failed.')

    w_0, gam, offset = opt_result.x

    return w_0, gam, offset

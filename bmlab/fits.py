import numpy as np
from scipy.optimize import minimize


class FitError(Exception):
    pass


def lorentz(w, w_0, gam):
    return 1. / ((w**2 - w_0**2)**2 + gam**2 * w_0**2)


def fit_lorentz(w, y):
    w_0_guess = (w[-1] + w[0]) / 2.
    gam_guess = (w[-1] - w[0]) / 4.

    def error(x, w, y): return np.sum((y - lorentz(w, *x))**2) / np.sum(y)

    opt_result = minimize(error, x0=(
        w_0_guess, gam_guess), tol=1.E-8, args=(w, y))
    if not opt_result.success:
        raise FitError('Lorentz fit failed.')

    w_0, gam = opt_result.x

    return w_0, gam

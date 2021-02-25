import numpy as np
from scipy.optimize import minimize
#import matplotlib.pyplot as plt

class FitError(Exception):
    pass


def lorentz(w, w_0, gam):
    return 1. / ((w**2 - w_0**2)**2 + gam**2 * w_0**2)


def fit_lorentz(w, y):
    w_0_guess = (w[-1] + w[0]) / 2.
    gam_guess = (w[-1] - w[0]) / 4.

    error = lambda x, w, y: np.sum( (y - lorentz(w, *x) )**2 ) / np.sum(y)

    opt_result = minimize(error, x0=(w_0_guess, gam_guess), method='BFGS', args=(w, y))
    if not opt_result.success:
        raise FitError('Lorentz fit failed.')

    w_0, gam = opt_result.x

    #plt.plot(w, y, 'o')
    #plt.plot(w, lorentz(w, w_0, gam))
    #plt.show()
    return w_0, gam

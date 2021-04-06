import logging

import numpy as np
from scipy.optimize import least_squares, minimize


logger = logging.getLogger(__name__)


class FitError(Exception):
    pass


def lorentz(w, w_0, gam, offset):
    return 1. / ((w**2 - w_0**2)**2 + gam**2 * w_0**2) + offset


def fit_lorentz(w, y):
    w_0_guess = w[np.argmax(y)]
    offset_guess = (y[0] + y[-1]) / 2.
    gam_guess = (w_0_guess ** 2 * (np.max(y) - offset_guess)) ** -0.5

    def error(x, w, y): return np.sum((y - lorentz(w, *x)) ** 2)

    opt_result = least_squares(error, x0=(
        w_0_guess, gam_guess, offset_guess),
        args=(w, y))

    if not opt_result.success:
        raise FitError('Lorentz fit failed.')

    w_0, gam, offset = opt_result.x

    return w_0, gam, offset


def fit_double_lorentz(w, y):
    w_0_guess = w[np.argmax(y)]
    offset_guess = (y[0] + y[-1]) / 2.
    gam_guess = (w_0_guess ** 2 * (np.max(y) - offset_guess)) ** -0.5
    n = len(w)
    w_0_guess_left = w[n // 3]
    w_0_guess_right = w[2 * n // 3]

    def error(x, w, y):
        return np.sum(
            (y - lorentz(w, *x[0:2], 0) - lorentz(w, *x[2:4], 0) - x[4])**2
        )

    opt_result = minimize(error, x0=(
        w_0_guess_left, gam_guess,
        w_0_guess_right, gam_guess, offset_guess),
        args=(w, y))

    if not opt_result.success:
        raise FitError('Lorentz fit failed.')

    return (*opt_result.x[0:2], opt_result.x[4]), \
           (*opt_result.x[2:4], opt_result.x[4])


def fit_circle(points):
    """
    Fits a circle to a given set of points. Returnes the center and the radius
    of the cricle. The inital parameters for the fiting process are arbitrarily
    chosen.

    Parameters
    ----------
    points: list of tuples

    Returns
    -------
    center_opt: tuple
                Coordinates of the circle center (x,y)
    radius_opt: float
                Radius of the circle
    """

    center, radius = calculate_exact_circle(points)

    param_guess = [center[0], center[1], radius]
    x_coords = np.array([xy[0] for xy in points])
    y_coords = np.array([xy[1] for xy in points])
    bnds = ((None, None), (None, None), (0.0, None))
    opt_result = minimize(_circle_opt,
                          param_guess,
                          args=(x_coords, y_coords),
                          bounds=bnds)

    return (opt_result['x'][0], opt_result['x'][1]), abs(opt_result['x'][2])


def _circle_opt(c, x_coord, y_coord):
    """
    Cost function to fit a circle to a given set of points.

    Parameters
    ----------
    c: array like
        Parameters to optimize: [x coordinates of circle center,
        y coordinates of circle center, radius of the cricle]
    x_coord:
        x coordinates of the points to fit
    y_coord
        y coordinates of the points to fit

    Returns
    -------
    out: float
        Cost value representing the deviation of the given set of points to
        the circle, which is represented by the parameter vector c.
    """
    return np.sum(
        ((x_coord - c[0]) ** 2
         + (y_coord - c[1]) ** 2
         - c[2] ** 2) ** 2)


def fit_spectral_region(region, xdata, ydata):
    mask = (region[0] < xdata) & (xdata < region[1])
    w0, gam, offset = fit_lorentz(xdata[mask], ydata[mask])
    logger.debug('Lorentz fit: w0 = %f, gam = %f, offset = %f' % (
        w0, gam, offset
    ))
    return gam, offset, w0


def calculate_exact_circle(points):
    # We only consider the first three points given
    if len(points) < 3:
        return

    x1 = points[0][0]
    y1 = points[0][1]
    x2 = points[1][0]
    y2 = points[1][1]
    x3 = points[2][0]
    y3 = points[2][1]

    x12 = x1 - x2
    x13 = x1 - x3

    y12 = y1 - y2
    y13 = y1 - y3

    sx13 = x1 ** 2 - x3 ** 2
    sy13 = y1 ** 2 - y3 ** 2

    sx21 = x2 ** 2 - x1 ** 2
    sy21 = y2 ** 2 - y1 ** 2

    x0 = -(sx13 * y12 +
           sy13 * y12 +
           sx21 * y13 +
           sy21 * y13) /\
        (2 * (x12 * y13 - x13 * y12))

    y0 = -(sx13 * x12 +
           sy13 * x12 +
           sx21 * x13 +
           sy21 * x13) /\
        (2 * (y12 * x13 - y13 * x12))

    c = (-x1 ** 2 - y1 ** 2 +
         2 * x0 * x1 + 2 * y0 * y1)

    r = np.sqrt(x0 ** 2 + y0 ** 2 - c)

    return (x0, y0), r

import logging

import numpy as np
from scipy.optimize import least_squares, minimize, fmin
from scipy.signal import find_peaks


logger = logging.getLogger(__name__)


class FitError(Exception):
    pass


def lorentz(x, w0, fwhm, intensity):
    with np.errstate(divide='ignore', invalid='ignore'):
        return intensity *\
               ((fwhm / 2) ** 2) / ((x - w0) ** 2 + (fwhm / 2) ** 2)


def fit_lorentz(x, y):
    w0_guess = float(x[np.argmax(y)])
    offset_guess = (y[0] + y[-1]) / 2.
    intensity_guess = np.max(y) - offset_guess
    fwhm_guess = float(2 * np.abs(
        w0_guess - x[np.argmax(y > (offset_guess + intensity_guess / 2))]))
    # Ensure a value larger than zero for the FWHM
    # (it might fail if the peak is not symmetric and its maximum is the
    # first value higher than offset_guess + intensity_guess/2)
    if fwhm_guess <= 0.0:
        fwhm_guess = 1.0

    def error(params, xdata, ydata):
        return (ydata
                - lorentz(xdata, *params[0:3])
                - params[3]) ** 2

    opt_result = least_squares(
        error,
        x0=(w0_guess, fwhm_guess, intensity_guess, offset_guess),
        args=(x, y)
    )

    if not opt_result.success:
        raise FitError('Lorentz fit failed.')

    w0, fwhm, intensity, offset = opt_result.x

    return w0, fwhm, intensity, offset


def fit_double_lorentz(x, y, bounds_w0=None):
    offset_guess = (y[0] + y[-1]) / 2.
    # gam_guess = (w0_guess ** 2 * (np.max(y) - offset_guess)) ** -0.5
    fwhm_guess = 4
    intensity_guess = np.max(y) - offset_guess

    # We run peak finding to get a good guess of the peak positions
    # and use the two peaks with highest prominence.
    peaks, properties = find_peaks(y, prominence=1)
    idx = np.argsort(properties['prominences'])[::-1]

    # Return if we didn't find two peaks
    if len(idx) < 2:
        return

    idx_sort = np.sort(peaks[idx[0:2]])
    w0_guess = list(x[idx_sort])

    def error(params, xdata, ydata):
        return (ydata
                - lorentz(xdata, *params[0:3])
                - lorentz(xdata, *params[3:6])
                - params[6]) ** 2

    # Create the bounds array
    if bounds_w0 is not None:
        # Lower limits
        bounds_lower = -np.Inf * np.ones(7)

        # 1st peak:
        # central position
        bounds_lower[0] = bounds_w0[0][0]
        # full-width-half-maximum
        # The VIPA spectrometer has an instrument width of
        # approx. 750 MHz, which translates to around
        # 7 px minimum peak width (FOB setup)
        # and 180 MHz and 4 px (780 nm setup).
        # We limit the minimal fit width to 3 px here.
        bounds_lower[1] = 3
        # intensity
        bounds_lower[2] = 0

        # 2nd peak:
        # central position
        bounds_lower[3] = bounds_w0[1][0]
        # full-width-half-maximum
        bounds_lower[4] = 3
        # intensity
        bounds_lower[5] = 0

        # offset
        bounds_lower[6] = 0

        # Upper limits
        bounds_upper = np.Inf * np.ones(7)

        bounds_upper[0] = bounds_w0[0][1]
        bounds_upper[3] = bounds_w0[1][1]
        bounds = (bounds_lower, bounds_upper)

        # Sort the guesses to the bounds
        bounds_w0_center = [np.mean(
            np.clip(bound, *x[::len(x) - 1])) for bound in bounds_w0]
        w0_guess.sort(reverse=(bounds_w0_center[0] > bounds_w0_center[1]))

        # Check that the initial guesses are within the bounds
        w0_guess = [np.clip(
            guess, *bounds_w0[idx]) for idx, guess in enumerate(w0_guess)]
    else:
        bounds = (-np.Inf, np.Inf)

    opt_result = least_squares(
        error,
        x0=(w0_guess[0], fwhm_guess, intensity_guess,
            w0_guess[1], fwhm_guess, intensity_guess,
            offset_guess
            ),
        args=(x, y),
        bounds=bounds
    )

    if not opt_result.success:
        raise FitError('Lorentz fit failed.')

    res = opt_result.x
    w0s, fwhms, intens = (res[0], res[3]), (res[1], res[4]), (res[2], res[5])
    offset = res[6]
    return w0s, fwhms, intens, offset


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

    # No need to fit if we only have three points
    if len(points) <= 3:
        return center, radius

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


def fit_lorentz_region(region, xdata, ydata, nr_peaks=1, bounds_w0=None):
    """
    Fits a lorentz or double lorentz fit to the given region

    Parameters
    ----------
    region: The section of the data to fit
    xdata: The x-data
    ydata: The y-data to fit
    nr_peaks: The number of peaks to fit
    bounds_w0: The bounds for the lorentz fit value of the maximum position

    Returns
    -------
    center, full-width-half-maximum, intensity and offset
    """
    try:
        if nr_peaks == 2:
            w0s, fwhms, intensities, offset = fit_double_lorentz(
                xdata[range(*region)], ydata[range(*region)],
                bounds_w0=bounds_w0)
        elif nr_peaks == 1:
            w0s, fwhms, intensities, offset = fit_lorentz(
                xdata[range(*region)], ydata[range(*region)])
        else:
            return
    except Exception:
        w0s = fwhms = intensities = offset = np.nan
    finally:
        return w0s, fwhms, intensities, offset


def calculate_exact_circle(points):
    # We need at least three points to fit a circle
    if len(points) < 3:
        return

    # If there are more than three points given,
    # we use the first and last and one in the center
    points = [points[idx]
              for idx in np.linspace(0, len(points) - 1, 3, dtype=int)]

    # If the first three points are all on the same line,
    # we slightly shift the first point so that we can
    # fit a circle
    if are_points_on_line(points):
        (x, y) = points[0]
        points[0] = (x - 1, y)

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


def are_points_on_line(points):
    """
    This function checks if the first three points in the given
    list lay on the same line
    Parameters
    ----------
    points: list of tuples representing points

    Returns
    -------
    boolean
    Whether the given points lay on the same line
    """
    if len(points) < 3:
        return True
    # We calculate a line form the first two points
    # and check whether the remaining points lay on it
    m = (points[0][1] - points[1][1]) /\
        (points[0][0] - points[1][0])
    n = points[0][1] - m * points[0][0]

    for point in points[:3]:
        if point[1] != (m * point[0] + n):
            return False

    return True


def fit_vipa(peaks, setup):
    """
    Fits the VIPA frequency axis

    Parameters
    ----------
    peaks: The peak positions
    setup: The setup parameters

    Returns
    -------
    The fitted VIPA parameters
    """
    # Check that we were given enough peaks
    if peaks is None:
        return
    if len(peaks) < (setup.calibration.num_brillouin_samples * 2 + 2):
        return

    # Calculate the start parameters for the VIPA fit
    r0 = peaks[0] * setup.pixel_size

    v1 = (setup.VIPA_PARAMS[0] +
          setup.VIPA_PARAMS[1] * r0 +
          setup.VIPA_PARAMS[2] * r0 ** 2) /\
         ((setup.vipa.m + setup.vipa.order) * np.pi)
    v2 = (setup.VIPA_PARAMS[1] +
          setup.VIPA_PARAMS[2] * 2 * r0) /\
         ((setup.vipa.m + setup.vipa.order) * np.pi) *\
        setup.pixel_size
    v3 = setup.VIPA_PARAMS[2] /\
        ((setup.vipa.m + setup.vipa.order) * np.pi) *\
        setup.pixel_size ** 2
    vipa_start = np.array([v1, v2, v3, setup.vipa.FSR])

    def error(vipa_params, peaks1, shifts1):
        fsr = vipa_params[3]
        frequencies = VIPA(peaks1, vipa_params) - setup.f0

        # Should match the expected frequencies
        d1 = frequencies - shifts1 - fsr * setup.calibration.orders

        # Should give equal values for Stokes and Anti-Stokes
        d2 = np.array([
            (frequencies[1] - frequencies[0]) -
            (frequencies[-1] - frequencies[-2]),
            (frequencies[2] - frequencies[0]) -
            (frequencies[-1] - frequencies[-3])
        ])

        # Should match the Brillouin shifts
        d3 = np.array([
            shifts1[1] - frequencies[1] + frequencies[0],
            shifts1[2] - frequencies[2] + frequencies[0],
            shifts1[2] - frequencies[5] + frequencies[3],
            shifts1[1] - frequencies[5] + frequencies[4]])

        return np.sum(d1 ** 2) + np.sum(d2 ** 2) + np.sum(d3 ** 2)

    opt_result = fmin(error,
                      vipa_start,
                      args=(peaks, setup.calibration.shifts))

    return opt_result[0], opt_result[1], opt_result[2], opt_result[3]


def VIPA(x, vipa_params):
    """
    Returns the absolute frequency in Hz of a given point
    on the spectrum in pixels.
    Subtract the absolute laser frequency from this value
    to get the relative shift.

    Parameters
    ----------
    x:   [pix]   The point on the spectrum
    vipa_params: Fit parameters of the VIPA fit

    Returns
    -------
    [Hz]   The frequency of the given point on the spectrum
    """
    return 1. / (vipa_params[0] + vipa_params[1] * x + vipa_params[2] * x ** 2)

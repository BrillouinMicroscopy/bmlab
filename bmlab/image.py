"""
Module to perform common image operations.
"""

import numpy as np
from scipy import optimize


def set_orientation(image, rotate=0, flip_ud=False, flip_lr=False):
    """
    Change the orientation of an image.

    The orientation of the raw images is setup dependent.

    Parameters
    ----------
    image: array_like
        Array of two dimensions.
    rotate: integer, default = 0
        Number of times the array is rotated by 90 degrees (clockwise).
    flip_ud: bool, default = False
        If True, flip the image up-down
    flip_lr: bool, default = False
        If True, flip the image left-right

    Returns
    -------
    image: array_like
        The input image in the requested orientation. The order
        of the operations is equal to the order
        of the input arguments: rotate, flip up-down, flip left right

    See Also
    --------
    numpy.rot90, numpy.flipud, numpy.fliplr
    """
    if not np.shape(np.shape(image)) == (2,):
        raise ValueError('Given Argument is not a two dimensional array')

    if rotate > 0:
        image = np.rot90(image, k=rotate, axes=(1, 0))

    if flip_ud:
        image = np.flipud(image)

    if flip_lr:
        image = np.fliplr(image)

    return image


def find_max_in_radius(img, xy0, radius):
    """
    Returns the index of the max. value in a circle around given point.

    Parameters
    ----------
    img: numpy.ndarray (2D)
        the image data
    xy0: tuple
        x-y indices of point around which to search for maximum
    radius: float
        the radius of the search

    Returns
    -------
    out: tuple
        x-y indices of the point of max. value
    """
    x, y = list(range(img.shape[0])), list(range(img.shape[1]))
    X, Y = np.meshgrid(x, y, indexing='ij')
    x0, y0 = xy0
    flat_img = np.nan * np.ones_like(img)
    mask = (X - x0) ** 2 + (Y - y0) ** 2 <= radius ** 2
    flat_img[mask] = img[mask]
    peak_idx = np.nanargmax(flat_img)
    peak_x, peak_y = np.unravel_index(peak_idx, img.shape, order='C')
    return peak_x, peak_y


def circle_opt(c, x_coord, y_coord):
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
    param_guess = [-50., -50., 500.]
    x_coords = np.array([xy[0] for xy in points])
    y_coords = np.array([xy[1] for xy in points])
    opt_result = optimize.minimize(circle_opt,
                                   param_guess,
                                   args=(x_coords, y_coords))

    return (opt_result['x'][0], opt_result['x'][1]), opt_result['x'][2]

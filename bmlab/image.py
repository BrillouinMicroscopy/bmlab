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


def calc_R(x, y, x_c, y_c):
    """
    Calculates the distances of a given set of points in a 2D space to a
    given center point.

    Parameters
    ----------
    x: array like
        x coordinates of the given points
    y: array like
        y coordinates of the given points
    xc: float
        x coordinate of the center
    yc: float
        y coordinate of the center

    Returns
    -------
    out: numpy array
    """
    return np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)


def f_opt(c, x, y):
    """
    Calculates the distance of a given set of points to a given center point
    and returns the deviaton to the mean distance.

    Parameters
    ----------
    c: parameter to be optimized, contains center coordinates (x,y)
    x: x coordinates of the given points
    y: y coordinates of the given points

    Returns
    -------
    out: numpy array containing the deviations of the calculated distances for
    given center coordinates to the mean distance
    """
    ri = calc_R(x, y, *c)
    return ri - ri.mean()


def fit_circle(points):
    """
    Fits a circle to a given set of points.Returnes the center and the radius
    of a given set of points.

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
    center_0 = (-250., -250.)
    x_coords = [xy[0] for xy in points]
    y_coords = [xy[1] for xy in points]
    opt_result = optimize.least_squares(f_opt,
                                        center_0, args=(x_coords, y_coords))
    center_opt = (opt_result['x'][0], opt_result['x'][1])
    radius_opt = calc_R(x_coords, y_coords, *center_opt).mean()
    return center_opt, radius_opt

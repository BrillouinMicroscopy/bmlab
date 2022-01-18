"""
Module to perform common image operations.
"""

import logging

import numpy as np
from scipy.interpolate import interpolate
import warnings


class AutofindException(Exception):
    pass


logger = logging.getLogger(__name__)


def set_orientation(image, rotate=0, flip_ud=False, flip_lr=False):
    """
    Change the orientation of an image.

    The orientation of the raw images is setup dependent.

    Parameters
    ----------
    image: array_like
        Array of two or three dimensions (containing multiple
        images along the first dimension in case of 3D)
    rotate: int
        Number of times the array is rotated by 90 degrees (clockwise).
    flip_ud: bool
        If True, flip the image up-down
    flip_lr: bool
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
    if np.shape(np.shape(image)) == (2,):
        if rotate > 0:
            image = np.rot90(image, k=rotate, axes=(1, 0))

        if flip_ud:
            image = np.flipud(image)

        if flip_lr:
            image = np.fliplr(image)
    elif np.shape(np.shape(image)) == (3,):
        if rotate > 0:
            image = np.rot90(image, k=rotate, axes=(2, 1))

        if flip_ud:
            image = image[:, ::-1, :]

        if flip_lr:
            image = image[:, :, ::-1]
    else:
        raise ValueError(
            'Given Argument is not a two or three dimensional array'
        )

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


def extract_lines_along_arc(img, arc):
    if arc.ndim != 3 or arc.shape[2] != 2:
        return

    m, n = img.shape
    func = interpolate.RegularGridInterpolator(
        (np.arange(m), np.arange(n)), img, method='linear', bounds_error=False)

    # In case the arc crosses the edge of the image, the interpolated array
    # will contain rows with only NaNs leading to an empty slice warning
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action='ignore',
            message='Mean of empty slice'
        )
        return np.nanmean(func((arc[:, :, 0], arc[:, :, 1])), 1)

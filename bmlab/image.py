"""
Module to perform common image operations.
"""

import logging

import numpy as np
from skimage.feature import blob_dog
from scipy.interpolate import interpolate

from bmlab.models.orientation import Orientation


class AutofindException(Exception):
    pass


logger = logging.getLogger(__name__)


def autofind_orientation(img):
    min_threshold = 1.E-6
    max_threshold = 0.01
    step_size = np.log10(2.)
    threshold = max_threshold
    max_tries = 10
    num_tries = 0

    while threshold > min_threshold:
        blobs = blob_dog(img, threshold=threshold)
        if len(blobs) > 6:
            threshold /= step_size
            step_size *= 1.5
            continue
        if len(blobs) > 4:
            break
        threshold *= step_size
        if num_tries > max_tries:
            raise AutofindException('Unable to find orientation')
        num_tries += 1

    nx, ny = img.shape
    nx2, ny2 = nx // 2, ny // 2

    blob_matrix = np.zeros((2, 2), dtype=int)
    for blob in blobs:
        x, y, _ = blob
        x = round(x)
        y = round(y)
        if x < nx2 and y < ny2:
            blob_matrix[0, 0] += 1
        elif x >= nx2 and y < ny2:
            blob_matrix[1, 0] += 1
        elif x < nx2 and y >= ny2:
            blob_matrix[0, 1] += 1
        else:
            blob_matrix[1, 1] += 1

    orient = Orientation()
    if np.all(blob_matrix == np.array([[2, 1], [1, 2]])):
        orient.set_reflection(vertically=True)
    elif np.all(blob_matrix == np.array([[1, 2], [2, 1]])):
        pass
    else:
        raise AutofindException('Unable to set orientation')

    logger.debug('Autofound orientation: %s' % orient)
    return orient


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

    return np.nanmean(func((arc[:, :, 0], arc[:, :, 1])), 1)

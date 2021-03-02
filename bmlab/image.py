"""
Module to perform common image operations.
"""

import logging

import numpy as np
from skimage.feature import blob_dog

from bmlab.model import Orientation


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

    blob_matrix = np.zeros((2, 2), dtype=np.int)
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


def extract_lines_along_arc(img, orientation, phis, circle, num_points):
    img = orientation.apply(img)
    values = []
    for phi in phis:
        e_r = circle.e_r(phi)
        mid_point = circle.point(phi)
        points = [mid_point + e_r *
                  k for k in np.arange(-num_points, num_points+1)]
        points = np.array(points)
        values.append(sum(interpolate(img, p) for p in points))
    return np.array(values)


def interpolate(img, xy):
    xy0 = np.array(xy, dtype=np.int)
    dxy = xy - xy0
    dxy = dxy.T
    ex = np.array([1, 0], dtype=np.int)
    ey = np.array([0, 1], dtype=np.int)
    nx, ny = img.shape

    if xy0[0] < 0 or xy0[0] >= nx - 1:
        return np.nan
    if xy0[1] < 0 or xy0[1] >= ny - 1:
        return np.nan
    res = img[tuple(xy0)] * (1 - dxy[0]) * (1 - dxy[1])
    res += img[tuple(xy0 + ex)] * dxy[0] * (1 - dxy[1])
    res += img[tuple(xy0 + ey)] * (1 - dxy[0]) * dxy[1]
    res += img[tuple(xy0 + ex + ey)] * dxy[0] * dxy[1]
    return res

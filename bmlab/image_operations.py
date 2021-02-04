"""
Module to rotate the raw images. The orientation
of the raw images is setup dependent.
"""

import numpy as np


def set_orientation(image, rotate=0, flip_ud=False, flip_lr=False):
    """
    Change the orientation of an image.

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

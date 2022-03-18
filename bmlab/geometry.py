import numpy as np
from shapely.geometry import Polygon, Point
from bmlab.utils import debug_timer


class Circle(object):
    """
    Representation of a circle.

    Wraps shapely-API for reduced complexity.
    """

    def __init__(self, center, radius):
        """
        Creates a circle instance.

        Parameters
        ----------
        center: 2-tuple
            Center of the circle.

        radius: float
            Radius of the circle.

        """
        self.center = np.array(center)
        self.radius = radius

        # Check that we have a valid circle
        # - no property is Inf or NaN
        # - radius is not zero
        self.valid = True
        if radius == 0 or\
            not np.isfinite(center).all() or\
                not np.isfinite(radius):
            self.valid = False

    def is_valid(self):
        return self.valid

    def point(self, phi, integer=False):
        """
        Returns the xy-coordinates of a point with given polar angle phi.

        Parameters
        ----------
        phi: float
            The polar angle (counted from 0-axis to given point).

        integer: bool
            Flag indicating whether the result should be integer
            pixel value or floating point values.

        Returns
        -------
        point: 2-tuple
            The xy-coordinates of the point.

        """
        if not self.valid:
            return None
        e_r = np.array([np.cos(phi), np.sin(phi)])
        pt = self.center + self.radius * e_r
        if integer:
            pt[0] = round(pt[0])
            pt[1] = round(pt[1])
            return np.array(pt, dtype=np.int)
        return pt

    def intersection(self, rect):
        """
        Calculates the intersection of the circle with a Rectangle object.

        Parameters
        ----------
        rect: Rectangle
            the rectangle

        Returns
        -------
        intersection: list of 2-tuples
            Points of intersection.
        """
        if not self.valid:
            return None
        center = Point(self.center)
        inter = center.buffer(self.radius).boundary.intersection(
            rect.poly).boundary
        return [(p.x, p.y) for p in inter.geoms]

    def angle(self, point):
        """
        Returns the polar angle for a given point.

        Parameters
        ----------
        point: 2-tuple
            The point for which to calculate the polar angle.

        Returns
        -------
        angle: float
            The polar angle.
        """
        if not self.valid:
            return None
        delta = np.array(point) - self.center
        if abs(delta[0]) < 1.E-9:
            if delta[1] > 0:
                return np.pi / 2.
            return 3 * np.pi / 2.
        return np.arctan(delta[1] / delta[0])

    def e_r(self, phi):
        return np.array([np.cos(phi), np.sin(phi)], dtype=float)


class Rectangle(object):
    def __init__(self, shape):
        self.poly = Polygon(
            [(0, 0), (0, shape[1]), (shape[0], shape[1]), (shape[0], 0)])


@debug_timer
def discretize_arc(circle, img_shape, num_points=200):
    """
    Returns a list of equidistant (in polar angle) points along a circle
    intersecting an image.

    Parameters
    ----------
    circle: Circle
        Circle that intersects the image.
    img_shape: 2-tuple
        Shape of the image.
    num_points: int
        How many points along the circle.

    Returns
    -------
    phis: numpy.ndarray
        polar angles of the discrete points on circular arc
    """
    rect = Rectangle(img_shape)
    cut_edges = circle.intersection(rect)
    if not cut_edges:
        return []
    phis_edges = [circle.angle(p) for p in cut_edges]
    if not phis_edges:
        return []
    return np.linspace(max(phis_edges), min(phis_edges), num_points)

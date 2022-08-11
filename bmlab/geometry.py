import numpy as np
import math
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
        e_r = self.e_r(phi)
        pt = self.center + self.radius * e_r
        if integer:
            pt[0] = round(pt[0])
            pt[1] = round(pt[1])
            return np.array(pt, dtype=np.int)
        return pt, e_r

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
        angle = math.atan2(delta[1], delta[0])
        if angle < 0:
            angle = angle + 2 * np.pi
        return angle

    @staticmethod
    def e_r(phi):
        return np.array([np.cos(phi), np.sin(phi)], dtype=float)


class Rectangle(object):
    def __init__(self, shape):
        self.poly = Polygon(
            [(0, 0), (0, shape[1]), (shape[0], shape[1]), (shape[0], 0)])


@debug_timer
def discretize_arc(circle, img_shape, num_points):
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
    if num_points is None:
        num_points = 2*(sum(np.square(img_shape)))**0.5
    rect = Rectangle(img_shape)
    cut_edges = circle.intersection(rect)
    cut_edges.sort(key=lambda edge: edge[0])
    if not cut_edges:
        return None
    phis_edges = [circle.angle(p) for p in cut_edges]
    if not phis_edges:
        return None
    return np.linspace(phis_edges[0], phis_edges[-1], num_points)

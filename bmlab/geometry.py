import numpy as np
import skimage.transform
from shapely.geometry import Polygon, Point


class Circle(object):

    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius

    def point(self, phi, integer=False):
        e_r = np.array([np.cos(phi), np.sin(phi)])
        pt = self.center + self.radius * e_r
        if integer:
            pt[0] = round(pt[0])
            pt[1] = round(pt[1])
            return np.array(pt, dtype=np.int)
        return pt

    def intersection(self, rect):
        center = Point(self.center)
        inter = center.buffer(self.radius).boundary.intersection(
            rect.poly).boundary
        return [(p.x, p.y) for p in inter]

    def angle(self, point):
        delta = np.array(point) - self.center
        if abs(delta[0]) < 1.E-9:
            if delta[1] > 0:
                return np.pi / 2.
            return 3 * np.pi / 2.
        return np.arctan(delta[1] / delta[0])

    def rect_mask(self, img_shape, phi, length, width):
        mask = np.zeros(img_shape, dtype=np.bool)
        pt = self.point(phi, integer=True)
        phi_degree = phi / np.pi * 180.
        mask[pt[0] - width // 2:pt[0] + width // 2,
             pt[1] - length // 2:pt[1] + length // 2] = True
        return skimage.transform.rotate(mask, phi_degree + 90,
                                        center=(pt[1], pt[0]))


class Rectangle(object):
    def __init__(self, shape):
        self.poly = Polygon(
            [(0, 0), (0, shape[1]), (shape[0], shape[1]), (shape[0], 0)])

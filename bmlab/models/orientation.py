from bmlab.serializer import Serializer


class Orientation(Serializer):

    def __init__(self, rotation=0,
                 reflection={'vertically': False, 'horizontally': False}):
        self.rotation = rotation
        self.reflection = reflection

    def set_rotation(self, num_rots):
        self.rotation = num_rots

    def set_reflection(self, **kwargs):
        axes = ['vertically', 'horizontally']
        for a in axes:
            if a in kwargs:
                self.reflection[a] = kwargs[a]

    def apply(self, img):
        from bmlab.image import set_orientation
        return set_orientation(img, self.rotation,
                               self.reflection['vertically'],
                               self.reflection['horizontally'])

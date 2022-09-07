import logging

from bmlab.models.regions import regions_merge_add_region, regions_check_order
from bmlab.serializer import Serializer

logger = logging.getLogger(__name__)


class PeakSelectionModel(Serializer):

    def __init__(self):
        self.brillouin_regions_f = []
        self.rayleigh_regions_f = []

    def add_brillouin_region(self, region):
        region = tuple(round(x) for x in region)

        regions_merge_add_region(
            self.brillouin_regions_f, region)

        regions_check_order(self.brillouin_regions_f)

    def set_brillouin_region(self, index, region):
        region = tuple(round(x) for x in region)

        if index >= len(self.brillouin_regions_f):
            self.add_brillouin_region(region)
        else:
            self.brillouin_regions_f[index] = region

        regions_check_order(self.brillouin_regions_f)

    def get_brillouin_regions(self):
        return self.brillouin_regions_f

    def clear_brillouin_regions(self):
        self.brillouin_regions_f = []

    def add_rayleigh_region(self, region):
        region = tuple(round(x) for x in region)

        regions_merge_add_region(
            self.rayleigh_regions_f, region)

        regions_check_order(self.rayleigh_regions_f)

    def set_rayleigh_region(self, index, region):
        region = tuple(round(x) for x in region)

        if index >= len(self.rayleigh_regions_f):
            self.add_rayleigh_region(region)
        else:
            self.rayleigh_regions_f[index] = region

        regions_check_order(self.rayleigh_regions_f)

    def get_rayleigh_regions(self):
        return self.rayleigh_regions_f

    def clear_rayleigh_regions(self):
        self.rayleigh_regions_f = []

import numpy as np


def regions_merge_add_region(regions, region):
    regions_fused = False

    # check if the selected regions overlap
    for i, saved_region in enumerate(regions):
        if (np.min(region) < np.max(saved_region)
                and (np.max(region) > np.min(saved_region))):
            # fuse overlapping regions
            regions[i] = (
                np.min([region, saved_region]),
                np.max([region, saved_region]))
            regions_fused = True

    if not regions_fused:
        regions.append(region)
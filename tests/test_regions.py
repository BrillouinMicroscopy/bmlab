from bmlab.models.regions import regions_merge_add_region


def test_merge_add_region():
    regions = [(1, 5), (6, 7)]

    regions_merge_add_region(regions, (8, 9))

    assert regions == [(1, 5), (6, 7), (8, 9)]

    regions_merge_add_region(regions, (8, 10))

    assert regions == [(1, 5), (6, 7), (8, 10)]

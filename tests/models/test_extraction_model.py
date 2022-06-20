import numpy as np

from bmlab.models.extraction_model import ExtractionModel
from bmlab.geometry import Circle


def test_add_and_set_points():
    em = ExtractionModel()
    em.add_point('0', 0, 0, 1)

    assert (0, 1) in em.get_points('0')

    em.set_point('0', 0, 0, 0, 2)
    em.set_point('0', 2, 0, 1, 2)

    assert em.get_points('0')[0] == (0, 2)
    assert em.get_points('0')[-1] == (1, 2)


def test_extraction_model_triggers_position_interpolation():

    em = ExtractionModel()

    em.set_image_shape((500, 500))
    em.add_point('0', 0, 0, 1)
    em.add_point('0', 0, 1, 0)

    assert (0, 1) in em.get_points('0')
    assert (1, 0) in em.get_points('0')
    np.testing.assert_equal(em.get_arc_by_time(0), np.empty(0))

    em.add_point('0', 0, 0, -1)

    assert (0, -1) in em.get_points('0')
    assert em.get_arc_by_time(0).shape == (500, 5, 2)


def test_clear_points_from_extraction_model():
    em = ExtractionModel()
    em.add_point('0', 0, 15, 15)
    em.add_point('0', 0, 75, 85)

    em.clear_points('0')
    assert em.get_points('0') == []
    assert em.get_time('0') == []


def test_set_arc_width():
    em = ExtractionModel()

    # Test default width
    assert em.arc_width == 2

    em.set_arc_width(3)

    assert em.arc_width == 3


def test_get_arc_by_calib_key():
    calib_keys = ['0', '1']
    times = [100, 500]
    circle_center = [(0, 0), (10, 0)]
    circle_radii = [9, 10]
    arc_start_end = [
        [[0.0, 9.0], [9.0, 0.0]],
        [[0.0, 0.0], [20.0, 0.0]]
    ]

    em = ExtractionModel()
    em.set_image_shape((100, 100))
    em.set_arc_width(2)
    # Set the circle points and test getting arc by calib_key
    for i, calib_key in enumerate(calib_keys):
        radius = circle_radii[i]

        circle = Circle(circle_center[i], radius)

        phis = [0, np.pi/4, np.pi/2]
        for phi in phis:
            (xdata, ydata), _ = circle.point(phi)
            em.add_point(calib_key, times[i], xdata, ydata)

        # Get the arc
        arc = em.get_arc_by_calib_key(calib_key)

        assert len(arc) == 500

        np.testing.assert_allclose(arc[0, 2, :], arc_start_end[i][0],
                                   rtol=0.05, atol=0.000001)
        np.testing.assert_allclose(arc[-1, 2, :], arc_start_end[i][1],
                                   rtol=0.05, atol=0.000001)


def test_get_arc_by_time():
    """
    Test getting arc by time
    """
    calib_keys = ['0', '1']
    times = [100, 500]
    circle_center = [(0, 0), (0, 1)]
    circle_radii = [10, 10]

    em = ExtractionModel()
    em.set_image_shape((100, 100))
    em.set_arc_width(2)
    # Set the circle points and test getting arc by calib_key
    for i, calib_key in enumerate(calib_keys):
        radius = circle_radii[i]

        circle = Circle(circle_center[i], radius)

        phis = [0, np.pi/4, np.pi/2]
        for phi in phis:
            (xdata, ydata), _ = circle.point(phi)
            em.add_point(calib_key, times[i], xdata, ydata)

    # the exact time point of a calibration should yield
    # the same as the calibration key
    arc = em.get_arc_by_time(times[0])
    expected_arc = em.get_arc_by_calib_key(calib_keys[0])
    np.testing.assert_allclose(arc, expected_arc,
                               rtol=0.05, atol=0.000001)

    # a time point after the last calibration should give
    # the value of the last calibration
    arc = em.get_arc_by_time(-500)
    expected_arc = em.get_arc_by_calib_key(calib_keys[0])
    np.testing.assert_allclose(arc, expected_arc,
                               rtol=0.05, atol=0.000001)
    arc = em.get_arc_by_time(times[-1] + 500)
    expected_arc = em.get_arc_by_calib_key(calib_keys[-1])
    np.testing.assert_allclose(arc, expected_arc,
                               rtol=0.05, atol=0.000001)

    # a time point between two calibrations should give
    # an interpolated value
    arc = em.get_arc_by_time(np.mean(times[0:2]))
    expected_arc_start = [0.0, 10.5]
    np.testing.assert_allclose(arc[0, 2, :], expected_arc_start,
                               rtol=0.05, atol=0.000001)


def test_get_arc_from_circle_phis():
    circle = Circle((0, 0), 100)
    phis = [0, np.pi/2, np.pi]
    arc_width = 2

    arc_expected = np.array([
        [[98.0, 0.0], [99, 0.0], [100.0, 0.0], [101.0, 0.0], [102.0, 0.0]],
        [[0.0, 98.0], [0.0, 99.0], [0.0, 100.0], [0.0, 101.0], [0.0, 102.0]],
        [[-98.0, 0.0], [-99, 0.0], [-100.0, 0.0], [-101.0, 0.0], [-102.0, 0.0]]
    ])

    arc = ExtractionModel().get_arc_from_circle_phis(circle, phis, arc_width)
    np.testing.assert_allclose(arc, arc_expected, atol=1e-12)

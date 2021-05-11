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


def test_extraction_model_triggers_circle_fit():

    em = ExtractionModel()
    em.add_point('0', 0, 0, 1)
    em.add_point('0', 0, 1, 0)

    assert (0, 1) in em.get_points('0')
    assert (1, 0) in em.get_points('0')
    assert em.get_circle_fit('0') is None

    em.add_point('0', 0, 0, -1)

    assert (0, -1) in em.get_points('0')
    cf = em.get_circle_fit('0')
    center, radius = cf

    np.testing.assert_allclose(center, (0, 0), atol=1.E-3)
    np.testing.assert_almost_equal(radius, 1, decimal=4)


def test_get_circle_fit_by_time():

    em = ExtractionModel()
    em.add_point('0', 10, 0, 1)
    em.add_point('0', 10, 1, 0)

    assert em.get_circle_fit_by_time(0) is None

    em.add_point('0', 10, 0, -1)

    # Get value before first entry
    center, radius = em.get_circle_fit_by_time(0)
    np.testing.assert_allclose(center, (0, 0), atol=1.E-3)
    np.testing.assert_almost_equal(radius, 1, decimal=4)

    # Exactly first entry
    center, radius = em.get_circle_fit_by_time(10)
    np.testing.assert_allclose(center, (0, 0), atol=1.E-3)
    np.testing.assert_almost_equal(radius, 1, decimal=4)

    # Get value after first entry
    center, radius = em.get_circle_fit_by_time(20)
    np.testing.assert_allclose(center, (0, 0), atol=1.E-3)
    np.testing.assert_almost_equal(radius, 1, decimal=4)

    em.add_point('1', 20, 0, 2)
    em.add_point('1', 20, 2, 0)
    em.add_point('1', 20, 0, -2)

    # Get value before first entry
    center, radius = em.get_circle_fit_by_time(0)
    np.testing.assert_allclose(center, (0, 0), atol=1.E-3)
    np.testing.assert_almost_equal(radius, 1, decimal=4)

    # Exactly first entry
    center, radius = em.get_circle_fit_by_time(10)
    np.testing.assert_allclose(center, (0, 0), atol=1.E-3)
    np.testing.assert_almost_equal(radius, 1, decimal=4)

    # Between first and second
    center, radius = em.get_circle_fit_by_time(15)
    np.testing.assert_allclose(center, (0, 0), atol=1.E-3)
    np.testing.assert_almost_equal(radius, 1.5, decimal=4)

    # Exactly second entry
    center, radius = em.get_circle_fit_by_time(20)
    np.testing.assert_allclose(center, (0, 0), atol=1.E-3)
    np.testing.assert_almost_equal(radius, 2, decimal=4)

    # Get value after last entry
    center, radius = em.get_circle_fit_by_time(30)
    np.testing.assert_allclose(center, (0, 0), atol=1.E-3)
    np.testing.assert_almost_equal(radius, 2, decimal=4)


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


def test_get_arc_by_x():
    calib_keys = ['0', '1', '2']
    times = [100, 500, 900]
    circle_center = [(0, 0), (10, 0), (10, 10)]
    circle_radii = [9, 10, 11]

    em = ExtractionModel()
    # Set the circle points and test getting arc by calib_key
    for i, calib_key in enumerate(calib_keys):
        radius = circle_radii[i]

        circle = Circle(circle_center[i], radius)

        phis = [0, np.pi/4, np.pi/2]
        for phi in phis:
            (xdata, ydata) = circle.point(phi)
            em.add_point(calib_key, times[i], xdata, ydata)

        # Create angles at which to calculate the arc
        em.extraction_angles[calib_key] = [0, np.pi/2]
        em.refresh_extraction_angles_interpolation()

        # Get the arc
        em.set_arc_width(2)
        arc = em.get_arc_by_calib_key(calib_key)

        expected_arc = [
            np.array(
                [
                    [radius-2, 0.0],
                    [radius-1, 0.0],
                    [radius, 0.0],
                    [radius+1, 0.0],
                    [radius+2, 0.0]
                ]
            ) + circle_center[i],
            np.array(
                [
                    [0.0, radius-2],
                    [0.0, radius-1],
                    [0.0, radius],
                    [0.0, radius+1],
                    [0.0, radius+2]
                ]
            ) + circle_center[i]
        ]

        assert len(arc) == len(expected_arc)

        np.testing.assert_allclose(arc, expected_arc,
                                   rtol=0.05, atol=0.000001)

    """
    Test getting arc by time
    """
    # the exact time point of a calibration should yield
    # the same as the calibration key
    arc = em.get_arc_by_time(times[0])
    expected_arc = em.get_arc_by_calib_key(calib_keys[0])
    np.testing.assert_allclose(arc, expected_arc,
                               rtol=0.05, atol=0.000001)

    # a time point after the last calibration should give
    # the value of the last calibration
    arc = em.get_arc_by_time(times[-1] + 500)
    expected_arc = em.get_arc_by_calib_key(calib_keys[-1])
    np.testing.assert_allclose(arc, expected_arc,
                               rtol=0.05, atol=0.000001)

    # a time point between two calibrations should give
    # an interpolated value
    arc = em.get_arc_by_time(np.mean(times[0:2]))
    radius_interpolated = np.mean(circle_radii[0:2])
    circle_center_interpolated = (5, 0)
    expected_arc = [
        np.array(
            [
                [radius_interpolated - 2, 0.0],
                [radius_interpolated - 1, 0.0],
                [radius_interpolated, 0.0],
                [radius_interpolated + 1, 0.0],
                [radius_interpolated + 2, 0.0]
            ]
        ) + circle_center_interpolated,
        np.array(
            [
                [0.0, radius_interpolated - 2],
                [0.0, radius_interpolated - 1],
                [0.0, radius_interpolated],
                [0.0, radius_interpolated + 1],
                [0.0, radius_interpolated + 2]
            ]
        ) + circle_center_interpolated
    ]
    np.testing.assert_allclose(arc, expected_arc,
                               rtol=0.05, atol=0.000001)

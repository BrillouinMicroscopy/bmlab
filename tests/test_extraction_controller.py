import numpy as np

from bmlab.controllers import ExtractionController
from bmlab.models import ExtractionModel


def test_add_point(mocker):
    mocker.patch('bmlab.session.Session.get_calibration_time',
                 return_value=0)

    em = ExtractionModel()
    mocker.patch('bmlab.session.Session.extraction_model', return_value=em)

    ec = ExtractionController()
    ec.add_point('0', (15, 15))

    points = em.get_points('0')

    assert len(points) == 1
    assert points[0] == (15, 15)

    ec.add_point('0', (20, 20))

    assert len(points) == 2
    assert points[0] == (15, 15)
    assert points[1] == (20, 20)


def test_set_point(mocker):
    mocker.patch('bmlab.session.Session.get_calibration_time',
                 return_value=0)

    em = ExtractionModel()
    mocker.patch('bmlab.session.Session.extraction_model', return_value=em)

    ec = ExtractionController()
    ec.add_point('0', (15, 15))

    points = em.get_points('0')

    assert len(points) == 1
    assert points[0] == (15, 15)

    ec.add_point('0', (20, 20))

    assert len(points) == 2
    assert points[0] == (15, 15)
    assert points[1] == (20, 20)

    ec.set_point('0', 0, (10, 10))

    assert len(points) == 2
    assert points[0] == (10, 10)
    assert points[1] == (20, 20)

    ec.set_point('0', 2, (30, 20))

    assert len(points) == 3
    assert points[0] == (10, 10)
    assert points[1] == (20, 20)
    assert points[2] == (30, 20)


def test_optimize_points(mocker):

    imgs = np.zeros((1, 100, 100), dtype=int)
    imgs[0, 19:22, 19:22] = 1
    imgs[0, 79:82, 79:82] = 1

    em = ExtractionModel()

    mocker.patch('bmlab.session.Session.extraction_model', return_value=em)
    mocker.patch('bmlab.session.Session.get_calibration_image',
                 return_value=imgs)
    mocker.patch('bmlab.session.Session.get_calibration_time',
                 return_value=0)

    ec = ExtractionController()
    ec.add_point('0', (15, 15))
    ec.add_point('0', (5, 85))

    opt_points = em.get_points('0')

    assert (15, 15) == opt_points[0]
    assert (5, 85) == opt_points[1]

    ec.set_point('0', 1, (75, 85))

    ec.optimize_points('0', radius=10)
    opt_points = em.get_points('0')

    assert (19, 20) == opt_points[0]
    assert (79, 80) == opt_points[1]


def test_distance_point_to_line():
    ec = ExtractionController()

    np.testing.assert_almost_equal(0, ec.distance_point_to_line(
        (0.5, 0.5), (1, 1), (0, 0)
    ))
    np.testing.assert_almost_equal(0.5, ec.distance_point_to_line(
        (0.5, 0.5), (0, 0), (0, 1)
    ))
    np.testing.assert_almost_equal(np.sqrt(0.5), ec.distance_point_to_line(
        (0, 1), (1, 1), (0, 0)
    ))


def test_find_points(mocker):

    imgs = np.ones((1, 400, 200), dtype=int)
    imgs[0, 19:22, 19:22] = 5
    imgs[0, 379:382, 19:22] = 5
    imgs[0, 19:22, 179:182] = 5
    imgs[0, 379:382, 179:182] = 5

    em = ExtractionModel()

    mocker.patch('bmlab.session.Session.extraction_model', return_value=em)
    mocker.patch('bmlab.session.Session.get_calibration_image',
                 return_value=imgs)
    mocker.patch('bmlab.session.Session.get_calibration_time',
                 return_value=0)

    ec = ExtractionController()

    ec.find_points('0', 1, 4, 10)
    opt_points = em.get_points('0')

    assert len(opt_points) == 2
    assert (19, 180) == opt_points[0]
    assert (379, 20) == opt_points[1]

import pathlib
import os

import h5py

from bmlab.session import Session
from bmlab.geometry import Circle, discretize_arc
from bmlab.models.calibration_model import FitSet, RayleighFit
from bmlab.serializer import serialize, deserialize


def data_file_path(file_name):
    return pathlib.Path(__file__).parent / 'data' / file_name


def test_serialize_and_deserialize_session():
    session = Session.get_instance()
    session.set_file(data_file_path('Water.h5'))

    session.orientation.set_reflection(vertically=True, horizontally=False)

    session.set_current_repetition('0')

    cal = session.current_repetition().calibration
    em = session.extraction_model()
    for calib_key in session.get_calib_keys():
        points = [(100, 290), (145, 255), (290, 110)]
        time = cal.get_time(calib_key)
        for p in points:
            em.add_point(calib_key, time, *p)
        imgs = cal.get_image(calib_key)
        img = imgs[0, ...]
        em.optimize_points(calib_key, img)
        em.optimize_points(calib_key, img)
        em.optimize_points(calib_key, img)

        circle_fit = em.get_circle_fit(calib_key)
        center, radius = circle_fit
        circle = Circle(center, radius)
        phis = discretize_arc(circle, img.shape, num_points=500)

        session.extraction_model().set_extraction_angles(calib_key, phis)

        assert em.get_points(calib_key) != points
        assert em.get_circle_fit(calib_key)
        assert em.get_extracted_values(calib_key) is None

        session.extract_calibration_spectrum(calib_key)

    session.save()

    session.clear()

    session.load(data_file_path('Water.session.h5'))

    assert session.orientation.rotation == 0
    assert session.orientation.reflection['vertically']

    assert not session.orientation.reflection['horizontally']

    os.remove(data_file_path('Water.session.h5'))


def test_serialize_fitset():

    fit_set = FitSet()
    fit = RayleighFit('1', 3, 4, 11., 12., 13., 14.)
    fit_set.add_fit(fit)

    with h5py.File('tmpsession.h5', 'w') as f:
        serialize(fit_set, f, 'fits')

    with h5py.File('tmpsession.h5', 'r') as f:
        actual = deserialize(
            FitSet, f['fits']
        )

    actual_fit = actual.get_fit('1', 3, 4)
    expected_fit = fit_set.get_fit('1', 3, 4)
    assert actual_fit.w0 == expected_fit.w0
    assert actual_fit.fwhm == expected_fit.fwhm
    assert actual_fit.intensity == expected_fit.intensity
    assert actual_fit.offset == expected_fit.offset

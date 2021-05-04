import pathlib
from pathlib import Path
import os
import shutil
import uuid

import h5py
import numpy as np
import pytest

from bmlab.session import Session
from bmlab.geometry import Circle, discretize_arc
from bmlab.models.calibration_model import FitSet, RayleighFit
from bmlab.models.extraction_model import CircleFit, ExtractionModel
from bmlab.serializer import Serializer


@pytest.fixture()
def tmp_dir():
    tmp_dir = 'tmp' + str(uuid.uuid4())
    os.mkdir(tmp_dir)
    os.chdir(tmp_dir)
    yield tmp_dir
    os.chdir('..')
    try:
        shutil.rmtree(tmp_dir)
    # Windows sometimes does not correctly close the HDF file. Then we have a
    # file access conflict.
    except Exception as e:
        print(e)


def data_file_path(file_name):
    return pathlib.Path(__file__).parent / 'data' / file_name


def test_serialize_session(tmp_dir):
    session = Session.get_instance()

    shutil.copy(data_file_path('Water.h5'), Path.cwd() / 'Water.h5')

    session.set_file('Water.h5')

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

        circle_fit = em.get_circle_fit(calib_key)
        center, radius = circle_fit
        circle = Circle(center, radius)
        phis = discretize_arc(circle, img.shape, num_points=500)

        session.extraction_model().set_extraction_angles(calib_key, phis)

        assert em.get_circle_fit(calib_key)
        assert em.get_extracted_values(calib_key) is None

        session.extract_calibration_spectrum(calib_key)

    session.save()

    with h5py.File('Water.session.h5', 'r') as f:
        assert 'session/extraction_models/0/points/1' in f


def test_serialize_fitset(tmp_dir):

    fit_set = FitSet()
    fit = RayleighFit('1', 3, 4, 11., 12., 13., 14.)
    fit_set.add_fit(fit)

    with h5py.File('tmpsession.h5', 'w') as f:
        fit_set.serialize(f, 'fits')

    with h5py.File('tmpsession.h5', 'r') as f:
        actual = FitSet.deserialize(f['fits'])

    actual_fit = actual.get_fit('1', 3, 4)
    expected_fit = fit_set.get_fit('1', 3, 4)
    assert actual_fit.w0 == expected_fit.w0
    assert actual_fit.fwhm == expected_fit.fwhm
    assert actual_fit.intensity == expected_fit.intensity
    assert actual_fit.offset == expected_fit.offset


def test_de_serialize_CircleFit(tmp_dir):

    cf = CircleFit(center=(1., 2.), radius=3.)

    with h5py.File(str(tmp_dir) + 'abc.h5', 'w') as f:
        cf.serialize(f, 'circle')

        assert isinstance(f['circle'], h5py.Group)
        assert f['circle'].attrs['type'] == \
               'bmlab.models.extraction_model.CircleFit'

    with h5py.File(str(tmp_dir) + 'abc.h5', 'r') as f:
        cf = CircleFit.deserialize(f['circle'])

        np.testing.assert_array_equal(cf.center, (1., 2.))
        assert cf.radius == 3.


def test_de_serialize_ExtractionModel(tmp_dir):

    em = ExtractionModel()
    for p in [(100, 290), (145, 255), (290, 110)]:
        em.add_point('the_calib_key', 0.1, p[0], p[1])

    cf = em.circle_fits.get('the_calib_key')

    with h5py.File(str(tmp_dir) + 'abc.h5', 'w') as f:
        em.serialize(f, 'the_extraction_model')

        assert isinstance(f['the_extraction_model'], h5py.Group)

    with h5py.File(str(tmp_dir) + 'abc.h5', 'r') as f:
        em_actual = Serializer.deserialize(f['the_extraction_model'])

        actual_radius = em_actual.circle_fits.get('the_calib_key').radius
        assert actual_radius == cf.radius
        np.testing.assert_array_equal(
            em_actual.circle_fits.get('the_calib_key').center, cf.center)

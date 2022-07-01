from pathlib import Path
import os
import shutil
import uuid

import h5py
import numpy as np
import pytest

from bmlab.session import Session
from bmlab.models.calibration_model import FitSet, RayleighFit
from bmlab.models.extraction_model import CircleFit
from bmlab.controllers import CalibrationController, ExtractionController


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


@pytest.fixture()
def session_file(tmp_dir):

    session = Session.get_instance()

    shutil.copy(data_file_path('Water.h5'), Path.cwd() / 'Water.h5')

    session.set_file('Water.h5')

    session.set_reflection(vertically=True, horizontally=False)

    session.set_current_repetition('0')

    ec = ExtractionController()
    cc = CalibrationController()

    em = session.extraction_model()
    cm = session.calibration_model()

    img = session.get_payload_image('0', 0)
    em.set_image_shape(img.shape)

    points = [(100, 290), (145, 255), (290, 110)]
    for calib_key in session.get_calib_keys():
        for p in points:
            ec.add_point(calib_key, p)

        assert em.get_arc_by_calib_key(calib_key) != np.empty(0)
        assert cm.get_spectra(calib_key) is None

        cc.extract_spectra(calib_key)

    session.save()

    session.clear()

    yield 'Water.session.h5'


def data_file_path(file_name):
    return Path(__file__).parent / 'data' / file_name


def test_serialize_session(tmp_dir):
    session = Session.get_instance()

    shutil.copy(data_file_path('Water.h5'), Path.cwd() / 'Water.h5')

    session.set_file('Water.h5')

    session.set_reflection(vertically=True, horizontally=False)

    session.set_current_repetition('0')

    ec = ExtractionController()
    cc = CalibrationController()

    em = session.extraction_model()
    cm = session.calibration_model()

    points = [(100, 290), (145, 255), (290, 110)]
    for calib_key in session.get_calib_keys():
        for p in points:
            ec.add_point(calib_key, p)

        assert em.get_arc_by_calib_key(calib_key) != np.empty(0)
        assert cm.get_spectra(calib_key) is None

        cc.extract_spectra(calib_key)

    session.save()

    with h5py.File('Water.session.h5', 'r') as f:
        assert 'session/extraction_models/0/points/1' in f
        assert f.attrs['version'].startswith('bmlab')

    session.clear()


def test_deserialize_session_file(session_file):

    session = Session.get_instance()
    session.set_file('Water.h5')

    em = session.extraction_model()
    cm = session.calibration_model()
    assert em
    assert em.calib_times['2'] == 62.542
    assert len(cm.spectra['1']) > 0
    assert len(em.positions['2']) > 0
    assert em.positions_interpolation is not None

    session.clear()


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

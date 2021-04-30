import pathlib
import os

import pytest

from bmlab.session import Session


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    yield

    # We remove the session file created by the test
    session_filename = data_file_path('Water.session.h5')
    if os.path.exists(session_filename):
        os.remove(session_filename)


def data_file_path(file_name):
    return pathlib.Path(__file__).parent / 'data' / file_name


def test_session_is_singleton():
    session = Session.get_instance()

    with pytest.raises(Exception):
        Session()

    assert session

    id_session = id(session)

    session = Session.get_instance()
    assert id(session) == id_session


def test_session_initializes():
    session = Session.get_instance()
    session.set_file(data_file_path('Water.h5'))

    assert len(session.extraction_models) == 1
    assert len(session.calibration_models) == 1
    assert '0' in session.calibration_models.keys()


def test_clear_session():
    # Arrange; set up session
    session = Session.get_instance()
    session.set_file(data_file_path('Water.h5'))
    session.orientation.rotation = 1
    session.set_current_repetition('0')

    # Act
    session.clear()

    # Assert
    assert session.file is None
    assert session.orientation.rotation == 0
    assert session.current_repetition() is None


def test_session_filename():
    session = Session.get_instance()

    assert session.get_session_file_name('Water.h5') == 'Water.session.h5'


def test_create_session_file():
    session = Session.get_instance()
    h5_file_path = data_file_path('Water.h5')
    session.set_file(h5_file_path)
    session_filename = session.get_session_file_name(h5_file_path)

    session.save()

    assert os.path.exists(session_filename)

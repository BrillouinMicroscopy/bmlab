import pathlib
import os

from bmlab.session import Session


def data_file_path(file_name):
    return pathlib.Path(__file__).parent / 'data' / file_name


def test_serialize_and_deserialize_session():
    session = Session.get_instance()
    session.set_file(data_file_path('Water.h5'))

    session.orientation.set_rotation(2)
    session.orientation.set_reflection(vertically=True, horizontally=True)

    session.save()

    session.orientation.set_reflection(vertically=False, horizontally=False)
    session.orientation.set_rotation(1)

    session.load(data_file_path('Water.session.h5'))

    assert session.orientation.rotation == 2
    assert session.orientation.reflection['vertically']
    assert session.orientation.reflection['horizontally']

    os.remove(data_file_path('Water.session.h5'))

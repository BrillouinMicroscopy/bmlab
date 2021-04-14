import pathlib

from bmlab.session import Session


def data_file_path(file_name):
    return pathlib.Path(__file__).parent / 'data' / file_name


def test_serialize_and_deserialize_session():
    session = Session.get_instance()
    session.set_file(data_file_path('Water.h5'))

    session.orientation.set_rotation(2)
    session.orientation.set_reflection(vertically=True, horizontally=True)

    tmp_file_name = 'tmp_session.h5'

    session.save_to_hdf(data_file_path(tmp_file_name))

    session.orientation.set_reflection(vertically=False, horizontally=False)
    session.orientation.set_rotation(1)

    session.load_from_hdf(data_file_path(tmp_file_name))

    assert session.orientation.rotation == 2
    assert session.orientation.reflection['vertically']
    assert session.orientation.reflection['horizontally']

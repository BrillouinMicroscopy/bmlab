"""
The serializer module contains functions that allow
serialization / deserialization of custom objects to/from HDF files.
"""

import importlib
import inspect
import re
import logging
import numbers


logger = logging.getLogger(__name__)


def serialize(obj, parent, as_name):
    """
    Saves a custom object to an HDF file or group.

    Parameters
    ----------
    obj: Python object
        The object to save to HDF

    parent: HDF group
        The group under which to store the object. Can be a HDF file group.

    as_name: str
        The name under which to store the object.

    """

    self_as_group = parent.create_group(as_name)

    if obj is None:
        self_as_group.attrs['type'] = 'None'
        return

    self_as_group.attrs['type'] = '%s.%s' % (obj.__class__.__module__,
                                             obj.__class__.__name__)

    if isinstance(obj, dict):
        properties = obj
    else:
        properties = dict(obj.__dict__)

    for name, value in properties.items():

        if callable(value):
            continue

        if is_scalar(value) or is_list_like(value):
            self_as_group.create_dataset(name, data=value)
        elif isinstance(value, dict):
            serialize(value, self_as_group, name)
        elif value is None:
            pass  # maybe better create an empty group?
        else:
            serialize(value, self_as_group, name)


def is_scalar(value):
    return isinstance(value, str) or isinstance(value, numbers.Number)


def is_list_like(value):
    return hasattr(value, '__len__') and not isinstance(value, dict)


def deserialize(cls, group):
    """
    Build up object of given class by reading given HDF group.

    Parameters
    ----------
    cls: Python 3 class
        The type of object to build.
    group: HDF group
        The HDF group containing the data for the given class.

    Returns
    -------
    Object of a class type specified in the attrs dict of the
    given group.
    """

    # Determine class of object to build:
    class_name = group.attrs.get('type')
    if not class_name:
        class_name = cls.__name__

    # Create object without calling the constructor:
    raw_object = _init_raw_object(class_name)

    # Each child item (group or dataset) of the given group
    # is either another class, a dict or simple data:
    for key, value in group.items():
        prop_full_class_name = value.attrs.get('type')
        if prop_full_class_name:
            prop_class = _class_from_full_class_name(prop_full_class_name)
            prop = deserialize(prop_class, value)
            if isinstance(raw_object, dict):
                raw_object[key] = prop
            else:
                raw_object.__dict__[key] = prop
        else:
            if isinstance(raw_object, dict):
                raw_object[key] = _unwrap(value)
            else:
                raw_object.__dict__[key] = _unwrap(value)

    return raw_object


def _init_raw_object(full_class_name):
    """
    Create an object without calling its constructor.

    Parameters
    ----------
    full_class_name: str
        fully qualified name of the class (package.classname)

    Returns
    -------
    Instance of the requested class with no properties
    initialized.
    """
    class_ = _class_from_full_class_name(full_class_name)
    return class_.__new__(class_)


def _class_from_full_class_name(full_class_name):

    pattern = re.compile('(.+)[.]([^.]+)')
    match = pattern.match(full_class_name)
    module_name = match[1]
    class_name = match[2]
    module = importlib.import_module(module_name)
    return _class_in_module(module, class_name)


def _class_in_module(module, class_name):

    for name, class_ in inspect.getmembers(module, inspect.isclass):
        if name == class_name:
            return class_


def _unwrap(value):

    if hasattr(value, '__len__'):
        return value[()]
    else:
        return value

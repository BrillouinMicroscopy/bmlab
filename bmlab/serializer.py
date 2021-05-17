"""
The serializer module contains functions that allow
serialization / deserialization of custom objects to/from HDF files.
"""

import importlib
import inspect
import re
import logging
import numbers
import builtins

import h5py


logger = logging.getLogger(__name__)


class Serializer(object):

    def serialize(self, parent, name, skip=[]):
        group = parent.create_group(name)

        group.attrs['type'] = '%s.%s' % (self.__class__.__module__,
                                         self.__class__.__name__)

        for var_name, var_value in self.__dict__.items():

            if callable(var_value):
                continue

            if var_name in skip:
                continue

            self.do_serialize(group, var_value, var_name)

    def serialize_dict(self, parent, dic, name):
        group = parent.create_group(name)
        for key, value in dic.items():
            self.serialize(group, key)

    def serialize_array(self, parent, array, name):
        parent.create_dataset(name, data=array)

    def serialize_string(self, parent, string, name):
        ds = parent.create_dataset(name, data=string,
                                   dtype=h5py.string_dtype())
        ds.attrs['type'] = 'builtins.str'

    def serialize_number(self, parent, number, name):
        parent.create_dataset(name, data=number)

    def do_serialize(self, group, value, name):
        if isinstance(value, dict):
            wrapped_dict = SerializableDict(value)
            wrapped_dict.serialize(group, name)
        elif isinstance(value, list):
            wrapped_list = SerializableList(value)
            wrapped_list.serialize(group, name)
        elif isinstance(value, tuple):
            wrapped_list = SerializableTuple(value)
            wrapped_list.serialize(group, name)
        elif isinstance(value, Serializer):
            value.serialize(group, name)
        elif is_list_like(value) and not isinstance(value, str):
            self.serialize_array(group, value, name)
        elif is_scalar(value):
            self.serialize_number(group, value, name)
        elif isinstance(value, str):
            self.serialize_string(group, value, name)
        elif value is None:
            pass
        else:
            raise Exception('Cannot serialize variable %s' % name)

    @classmethod
    def deserialize(cls, group):
        class_ = group.attrs.get('type')

        if class_ == 'builtins.list':
            instance = list()
            instance_handle = instance
            num_items = len(group.keys())
            for idx in range(num_items):
                value = group[str(idx)]
                instance.append(None)
                cls.do_deserialize(instance_handle, value, -1)
            return instance
        elif class_ == 'builtins.tuple':
            instance = list()
            instance_handle = instance
            num_items = len(group.keys())
            for idx in range(num_items):
                value = group[str(idx)]
                instance.append(None)
                cls.do_deserialize(instance_handle, value, -1)
            return tuple(instance)

        if class_ == 'builtins.dict':
            instance = dict()
            instance_handle = instance
        else:
            instance = _init_raw_object(class_)
            instance_handle = instance.__dict__

        for var_name, var_value in group.items():
            cls.do_deserialize(instance_handle, var_value, var_name)

        if isinstance(instance, Serializer):
            instance.post_deserialize()

        return instance

    @classmethod
    def do_deserialize(cls, instance_handle, var_value, var_name):
        if isinstance(var_value, h5py.Dataset):
            if var_value.shape == ():
                item = var_value[...].item()
                if isinstance(item, bytes):
                    item = item.decode('utf-8')
                instance_handle[var_name] = item
            else:
                instance_handle[var_name] = var_value[...]
        elif isinstance(var_value, h5py.Group):
            instance_handle[var_name] = Serializer.deserialize(var_value)
        else:
            raise Exception('Cannot deserialize object %s' % var_name)

    def post_deserialize(self):
        """
        Override this method in derived classes if further initialization
        is necessary after deserialization from HDF.
        """
        pass


class SerializableTuple(Serializer, builtins.tuple):

    def __init__(self, pure_tuple):
        self.pure_tuple = pure_tuple

    def serialize(self, parent, name):
        group = parent.create_group(name)
        group.attrs['type'] = 'builtins.tuple'
        for idx, value in enumerate(self.pure_tuple):
            self.do_serialize(group, value, str(idx))


class SerializableList(Serializer, builtins.list):

    def __init__(self, pure_list):
        self.pure_list = pure_list

    def serialize(self, parent, name):
        group = parent.create_group(name)
        group.attrs['type'] = 'builtins.list'
        for idx, value in enumerate(self.pure_list):
            self.do_serialize(group, value, str(idx))


class SerializableDict(Serializer, builtins.dict):

    def __init__(self, pure_dict):
        self.pure_dict = pure_dict

    def serialize(self, parent, name):
        group = parent.create_group(name)

        group.attrs['type'] = 'builtins.dict'

        for key, value in self.pure_dict.items():
            if callable(value):
                continue
            self.do_serialize(group, value, key)


def is_scalar(value):
    return isinstance(value, str) or isinstance(value, numbers.Number)


def is_list_like(value):
    return hasattr(value, '__len__') and not isinstance(value, dict)


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

"""
The serializer module contains classes and functions that allow
serialization / deserialization of custom objects to/from HDF files.

In order to enable this feature for a given class, add the ModelSerializerMixin
to the base class(es) of the given class.
"""

import importlib
import inspect
import re


class ModelSerializerMixin(object):
    """
    Mixin used for model classes to enable recursive serialization and
    deserialization to/from HDF files.
    """

    def serialize(self, parent, as_name=None):
        """
        Store the object in an HDF group.

        Parameters
        ----------
        parent: HDF group object
            The parent group in which to store the current object

        as_name: str
            The name under which to store the current object.
            Properties of the current object will be automatically stored
            under the name of the property.
        """
        if as_name is not None:
            self_as_group = parent.create_group(as_name)
        else:
            self_as_group = parent

        self_as_group.attrs['type'] = '%s.%s' % (self.__class__.__module__,
                                                 self.__class__.__name__)

        if isinstance(self, SerializableDict):
            vars = self
        else:
            vars = dict(self.__dict__)

        for var_name, var_value in vars.items():
            if isinstance(var_value, ModelSerializerMixin):
                var_value.serialize(self_as_group, var_name)
            elif isinstance(var_value, dict):
                wrapped = SerializableDict(var_value)
                wrapped.serialize(self_as_group, var_name)
            else:
                self_as_group.create_dataset(var_name, data=var_value)

    @classmethod
    def deserialize(cls, group):
        """
        Build up object of given class by reading given HDF group.

        Parameters
        ----------
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
        raw_object = init_raw_object(class_name)

        # Each child item (group or dataset) of the given group
        # is either another class, a dict or simple data:
        for key, value in group.items():
            prop_full_class_name = value.attrs.get('type')
            if prop_full_class_name:
                prop_class = class_from_full_class_name(prop_full_class_name)
                prop = prop_class.deserialize(value)
                if isinstance(raw_object, dict):
                    raw_object[key] = prop
                else:
                    raw_object.__dict__[key] = prop
            else:
                if isinstance(raw_object, dict):
                    raw_object[key] = unwrap(value)
                else:
                    raw_object.__dict__[key] = unwrap(value)

        return raw_object


class SerializableDict(dict, ModelSerializerMixin):
    """
    Wrapper around the built-in dict class.
    """

    def __init__(self, content):
        dict.__init__(self)
        for key, value in content.items():
            self[key] = value


def init_raw_object(full_class_name):
    class_ = class_from_full_class_name(full_class_name)
    return class_.__new__(class_)


def class_from_full_class_name(full_class_name):
    pattern = re.compile('(.+)[.]([^.]+)')
    match = pattern.match(full_class_name)
    module_name = match[1]
    class_name = match[2]
    module = importlib.import_module(module_name)
    return class_in_module(module, class_name)


def class_in_module(module, class_name):
    for name, class_ in inspect.getmembers(module, inspect.isclass):
        if name == class_name:
            return class_


def unwrap(value):
    if hasattr(value, '__len__'):
        return value[()]
    else:
        return value

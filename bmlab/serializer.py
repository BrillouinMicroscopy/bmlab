import importlib
import inspect
import re


class ModelSerializerMixin(object):

    @classmethod
    def serialize(cls, obj, parent, as_name=None):
        if as_name is not None:
            self_as_group = parent.create_group(as_name)
        else:
            self_as_group = parent

        self_as_group.attrs['type'] = '%s.%s' % (obj.__class__.__module__,
                                                 obj.__class__.__name__)

        if isinstance(obj, SerializableDict):
            vars = obj
        else:
            vars = dict(obj.__dict__)

        for var_name, var_value in vars.items():
            if isinstance(var_value, ModelSerializerMixin):
                var_value.serialize(self_as_group, var_name)
            elif isinstance(var_value, dict):
                wrapped = SerializableDict(var_value)
                wrapped.serialize(wrapped, self_as_group, var_name)
            else:
                self_as_group.create_dataset(var_name, data=var_value)

    @classmethod
    def deserialize(cls, group):
        class_name = group.attrs.get('type')
        if not class_name:
            class_name = cls.__name__
        raw_object = init_raw_object(class_name)
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
                    raw_object[key] = cls.unwrap(value)
                else:
                    raw_object.__dict__[key] = cls.unwrap(value)

        return raw_object

    @classmethod
    def unwrap(cls, value):
        if hasattr(value, '__len__'):
            return value[()]
        else:
            return value


class SerializableDict(dict, ModelSerializerMixin):

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

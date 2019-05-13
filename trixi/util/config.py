#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from copy import deepcopy

from trixi.util.util import ModuleMultiTypeDecoder, ModuleMultiTypeEncoder


class Config(dict):
    """
    Config is the main object used to store configurations. As a rule of thumb, anything you might
    want to change in your experiment should go into the Config. It's basically a :class:`dict`,
    but vastly more powerful. Key features are

        - Access keys as attributes
            Config["a"]["b"]["c"] is the same as Config.a.b.c.
            Can also be used for setting if the second to last key exists. Only works for keys that
            conform with Python syntax (Config.myattr-1 is not allowed).
        - Advanced de-/serialization
            Using specialized JSON encoders and decoders, almost anything can be serialized and
            deserialized. This includes types, functions (except lambdas) and modules. For example,
            you could have something like::

                c = Config(model=MyModel)
                c.dump("somewhere")

            and end up with a JSON file that looks like this::

                {
                    "model": "__type__(your.model.module.MyModel)"
                }

            and vice versa. We use double underscores and parentheses for serialization,
            so it's probably a good idea to not use this pattern for other stuff!
        - Automatic CLI exposure
            If desired, the Config will create an ArgumentParser that contains all keys in the
            Config as arguments in the form "- - key", so you can run your experiment from the command
            line and manually overwrite certain values. Deeper levels are also accessible via dot
            notation "- - key_with_dict_value.inner_key".
        - Comparison
            Compare any number of Configs and get a new Config containing only the values that
            differ among input Configs.

    Args:
        file_ (str): Load Config from this file.
        config (Config): Update with values from this Config (can be combined with :attr:`file_`).
            Will by default only make shallow copies, see :attr:`deep`.
        update_from_argv (bool): Update values from argv. Will automatically expose keys to the
            CLI as '- - key'.
        deep (bool): Make deep copies if :attr:`config` is given.

    """

    def __init__(self, file_=None, config=None, update_from_argv=False, deep=False, **kwargs):

        super(Config, self).__init__()

        # the following allows us to access keys as attributes if syntax permits
        # config["a"] = 1 -> config.a -> 1
        # config["a-b"] = 2 -> config.a-b (not possible)
        # this is purely for convenience
        self.__dict__ = self

        if file_ is not None:
            self.load(file_)

        if config is not None:
            if deep:
                # convert config to Config (in case it's just a dict)
                # and get deepcopy
                config = Config(config=config, update_from_argv=False, deep=False)
                config = config.deepcopy()
            self.update(config, deep=False)

        if len(kwargs) >= 1:
            if deep:
                kwargs = Config(config=kwargs, update_from_argv=False, deep=False)
                kwargs = kwargs.deepcopy()
            self.update(kwargs, deep=False)

        if update_from_argv:
            update_from_sys_argv(self)

    def update(self, dict_like, deep=False, ignore=None, allow_dict_overwrite=True):
        """Update entries in the Config.

        Args:
            dict_like (dict or derivative thereof): Update source.
            deep (bool): Make deep copies of all references in the source.
            ignore (iterable): Iterable of keys to ignore in update.
            allow_dict_overwrite (bool): Allow overwriting with dict.
                Regular dicts only update on the highest level while we recurse
                and merge Configs. This flag decides whether it is possible to
                overwrite a 'regular' value with a dict/Config at lower levels.
                See examples for an illustration of the difference


        Examples:
            The following illustrates the update behaviour if
            :obj:allow_dict_overwrite is active. If it isn't, an AttributeError
            would be raised, originating from trying to update "string"::

                config1 = Config(config={
                    "lvl0": {
                        "lvl1": "string",
                        "something": "else"
                    }
                })

                config2 = Config(config={
                    "lvl0": {
                        "lvl1": {
                            "lvl2": "string"
                        }
                    }
                })

                config1.update(config2, allow_dict_overwrite=True)

                >>>config1
                {
                    "lvl0": {
                        "lvl1": {
                            "lvl2": "string"
                        },
                        "something": "else"
                    }
                }

        """

        if ignore is None:
            ignore = ()

        if deep:
            update_config = Config(config=dict_like, deep=True)
            self.update(update_config,
                        deep=False,
                        ignore=ignore,
                        allow_dict_overwrite=allow_dict_overwrite)

        else:
            for key, value in dict_like.items():
                if key in ignore:
                    continue
                if key in self and isinstance(value, dict):
                    try:
                        self[key].update(value,
                                         deep=False,
                                         ignore=ignore,
                                         allow_dict_overwrite=allow_dict_overwrite)
                    except AttributeError as ae:
                        if allow_dict_overwrite:
                            self[key] = value
                        else:
                            raise ae
                else:
                    self[key] = value

    def deepupdate(self, dict_like, ignore=None, allow_dict_overwrite=True):
        """Identical to :meth:`update` with `deep=True`.

        Args:
            dict_like (dict or derivative thereof): Update source.
            ignore (iterable): Iterable of keys to ignore in update.
            allow_dict_overwrite (bool): Allow overwriting with dict.
                Regular dicts only update on the highest level while we recurse
                and merge Configs. This flag decides whether it is possible to
                overwrite a 'regular' value with a dict/Config at lower levels.
                See examples for an illustration of the difference

        """
        self.update(dict_like,
                    deep=True,
                    ignore=ignore,
                    allow_dict_overwrite=allow_dict_overwrite)

    def __setattr__(self, key, value):
        """Modified to automatically convert `dict` to Config."""

        if type(value) == dict:
            new_config = Config()
            new_config.update(value, deep=False)
            super(Config, self).__setattr__(key, new_config)
        else:
            super(Config, self).__setattr__(key, value)

    def __getitem__(self, key):
        """Allows convenience access to deeper levels using dots to separate
        levels, for example `config["a.b.c"]`.
        """

        if key == "":
            if len(self.keys()) == 1:
                key = list(self.keys())[0]
            else:
                raise KeyError("Empty string only works for single element Configs.")

        if type(key) == str and "." in key:
            superkey = key.split(".")[0]
            subkeys = ".".join(key.split(".")[1:])
            if superkey not in self:
                # this part enables ints in the access chain, e.g. a.1.b
                try:
                    intkey = int(superkey)
                    if intkey in self:
                        superkey = intkey
                except ValueError:
                    # if we can't convert to int, just continue so a KeyError will be raised
                    pass
            if type(self[superkey]) in (list, tuple):
                try:
                    subkeys = int(subkeys)
                except ValueError:
                    pass
            return self[superkey][subkeys]
        else:
            return super(Config, self).__getitem__(key)

    def __setitem__(self, key, value):
        """Allows convenience access to deeper levels using dots to separate
        levels, for example `config["a.b.c"]`.
        """

        if key == "":
            if len(self.keys()) == 1:
                key = list(self.keys())[0]
            else:
                raise KeyError("Empty string only works for single element Configs.")

        if type(key) == str and "." in key:
            superkey = key.split(".")[0]
            subkeys = ".".join(key.split(".")[1:])
            if superkey != "" and superkey not in self:
                self[superkey] = Config()
            if type(self[superkey]) == list:
                try:
                    subkeys = int(subkeys)
                except ValueError:
                    pass
            self[superkey][subkeys] = value
        elif type(value) == dict:
            super(Config, self).__setitem__(key, Config(config=value))
        else:
            super(Config, self).__setitem__(key, value)

    def set_with_decode(self, key, value, stringify_value=False):
        """Set single value, using :class:`.ModuleMultiTypeDecoder` to interpret
        key and value strings by creating a temporary JSON string.

        Args:
            key (str): Config key.
            value (str): New value key will map to.
            stringify_value (bool): If `True`, will insert the value into the
                temporary JSON as a real string. See examples!

        Examples:
            Example for when you need to set `stringify_value=True`::

                config.set_with_decode("key", "__type__(trixi.util.config.Config)", stringify_value=True)

            Example for when you need to set `stringify_value=False`::

                config.set_with_decode("key", "[1, 2, 3]")

        """

        if type(key) != str:
            # We could encode the key if it's not a string, but for now raise
            raise TypeError("set_with_decode requires string as key.")
        if type(value) != str:
            raise TypeError("set_with_decode requires string as value.")

        dict_str = ''
        depth = 0
        key_split = key.split(".")

        for k in key_split:
            dict_str += "{"
            dict_str += '"{}":'.format(k)
            depth += 1

        if stringify_value:
            dict_str += '"{}"'.format(value)
        else:
            dict_str += "{}".format(value)

        for _ in range(depth):
            dict_str += "}"

        self.loads(dict_str)

    def set_from_string(self, str_, stringify_value=False):
        """Set a value from a single string, separated with "=".
        Uses :meth:´set_with_decode´.

        Args:
            str_ (str): String that looks like "key=value".

        """

        key, value = str_.split("=")
        self.set_with_decode(key, value, stringify_value)

    def update_missing(self, dict_like, deep=False, ignore=None):
        """Recursively insert values that do not yet exist.

        Args:
            dict_like (dict or derivative thereof): Update source.
            deep (bool): Make deep copies of all references in the source.
            ignore (iterable): Iterable of keys to ignore in update.

        """

        for key, value in dict_like.items():

            if key not in self:
                if type(value) == Config:
                    if deep:
                        self[key] = value.deepcopy()
                    else:
                        self[key] = value
                else:
                    if deep:
                        self[key] = deepcopy(value)
                    else:
                        self[key] = value
            else:
                if isinstance(value, dict) and isinstance(self[key], dict):
                    self[key].update_missing(Config(config=value, deep=deep))

    def dump(self, file_, indent=4, separators=(",", ": "), **kwargs):
        """Write config to file using :meth:`json.dump`.

        Args:
            file_ (str or File): Write to this location.
            indent (int): Formatting option.
            separators (iterable): Formatting option.
            **kwargs: Will be passed to :meth:`json.dump`.

        """

        if hasattr(file_, "write"):
            json.dump(self, file_,
                      cls=ModuleMultiTypeEncoder,
                      indent=indent,
                      separators=separators,
                      **kwargs)
        else:
            with open(file_, "w") as file_object:
                json.dump(self, file_object,
                          cls=ModuleMultiTypeEncoder,
                          indent=indent,
                          separators=separators,
                          **kwargs)

    def dumps(self, indent=4, separators=(",", ": "), **kwargs):
        """Get string representation using :meth:`json.dumps`.

        Args:
            indent (int): Formatting option.
            separators (iterable): Formatting option.
            **kwargs: Will be passed to :meth:`json.dumps`.

        """
        return json.dumps(self,
                          cls=ModuleMultiTypeEncoder,
                          indent=indent,
                          separators=separators,
                          **kwargs)

    def load(self, file_, raise_=True, decoder_cls_=ModuleMultiTypeDecoder, **kwargs):
        """Load config from file using :meth:`json.load`.

        Args:
            file_ (str or File): Read from this location.
            raise (bool): Raise errors.
            decoder_cls_ (type): Class that is used to decode JSON string.
            **kwargs: Will be passed to :meth:`json.load`.

        """

        try:
            if hasattr(file_, "read"):
                new_dict = json.load(file_, cls=decoder_cls_, **kwargs)
            else:
                with open(file_, "r") as file_object:
                    new_dict = json.load(file_object, cls=decoder_cls_, **kwargs)
        except Exception as e:
            if raise_:
                raise e

        self.update(new_dict)

    def loads(self, json_str, decoder_cls_=ModuleMultiTypeDecoder, **kwargs):
        """Load config from JSON string using :meth:`json.loads`.

        Args:
            json_str (str): Interpret this string.
            decoder_cls_ (type): Class that is used to decode JSON string.
            **kwargs: Will be passed to :meth:`json.loads`.

        """

        if not json_str.startswith("{"):
            json_str = "{" + json_str
        if not json_str.endswith("}"):
            json_str = json_str + "}"
        new_dict = json.loads(json_str, cls=decoder_cls_, **kwargs)
        self.update(new_dict)

    def hasattr_not_none(self, key):
        try:
            result = self[key]
            return result is not None
        except KeyError as ke:
            return False

    def contains(self, dict_like):
        """Check whether all items in a dictionary-like object match the ones in
        this Config.

        Args:
            dict_like (dict or derivative thereof): Returns True if this is
                contained in this Config.

        Returns:
            bool: True if dict_like is contained in self, otherwise False.

        """

        dict_like_config = Config(config=dict_like)

        for key, val in dict_like_config.items():

            if key not in self:
                return False
            else:
                if isinstance(val, dict):
                    if not self[key].contains(val):
                        return False
                else:
                    if not self[key] == val:
                        return False

        return True

    def deepcopy(self):
        """Get a deep copy of this Config.

        Returns:
            Config: A deep copy of self.

        """

        def _deepcopy(source, target):
            for key, val in source.items():
                if not isinstance(val, dict):
                    try:
                        target[key] = deepcopy(val)
                    except TypeError as e:
                        target[key] = val
                else:
                    target[key] = Config()
                    _deepcopy(source[key], target[key])

        new_config = Config()
        _deepcopy(self, new_config)

        return new_config

    @staticmethod
    def init_objects(config):
        """Returns a new Config with types converted to instances.

        Any value that is a Config and contains a type key will be converted to
        an instance of that type::

            {
                "stuff": "also_stuff",
                "convert_me": {
                    type: {
                        "param": 1,
                        "other_param": 2
                    },
                    "something_else": "hopefully_useless"
                }
            }

        becomes::

            {
                "stuff": "also_stuff",
                "convert_me": type(param=1, other_param=2)
            }

        Note that additional entries can be lost as shown above.

        Args:
            config (Config): New Config will be built from this one

        Returns:
            Config: A new config with instances made from type entries.
        """

        def init_sub_objects(objs):
            if isinstance(objs, dict):
                ret_dict = Config()
                for key, val in objs.items():
                    if isinstance(key, type):
                        init_param = init_sub_objects(val)
                        if isinstance(init_param, dict):
                            init_obj = key(**init_param)
                        elif isinstance(init_param, (list, tuple, set)):
                            init_obj = key(*init_param)
                        else:
                            init_obj = key()
                        return init_obj
                    elif isinstance(val, (dict, list, tuple, set)):
                        ret_dict[key] = init_sub_objects(val)
                    else:
                        ret_dict[key] = val
                return ret_dict
            elif isinstance(objs, (list, tuple, set)):
                orig_type = type(objs)
                ret_list = []
                for el in objs:
                    ret_list.append(init_sub_objects(el))
                return orig_type(ret_list)
            else:
                return objs

        return init_sub_objects(config)

    def __str__(self):
        return self.dumps(sort_keys=True)

    def difference_config(self, *other_configs):
        """Get the difference of this and any number of other configs.
        See :meth:`difference_config_static` for more information.

        Args:
            *other_configs (Config): Compare these configs and self.

        Returns:
            Config: Difference of self and the other configs.

        """
        return self.difference_config_static(self, *other_configs)

    @staticmethod
    def difference_config_static(*configs, only_set=False):
        """Make a Config of all elements that differ between N configs.

        The resulting Config looks like this::

            {
                key: (config1[key], config2[key], ...)
            }

        If the key is missing, None will be inserted. The inputs will not be
        modified.

        Args:
            configs (Config): Any number of Configs
            only_set (bool): If only the set of different values hould be returned or for each config the
            corresponding one

        Returns:
            Config: Possibly empty Config
        """

        difference = dict()
        mmte = ModuleMultiTypeEncoder()

        all_keys = set()
        for config in configs:
            all_keys.update(set(config.keys()))

        for key in all_keys:

            current_values = []
            all_equal = True
            all_configs = True

            for config in configs:

                if key not in config:
                    all_equal = False
                    all_configs = False
                    current_values.append(None)
                else:
                    current_values.append(mmte._encode(config[key]))

                if len(current_values) >= 2:
                    if current_values[-1] != current_values[-2]:
                        all_equal = False

                if type(current_values[-1]) != Config:
                    all_configs = False

            if not all_equal:

                if not all_configs:
                    if not only_set:
                        difference[key] = tuple(current_values)
                    else:
                        difference[key] = tuple(set(current_values))
                else:
                    difference[key] = Config.difference_config_static(*current_values, only_set=only_set)

        return Config(config=difference)

    def flat(self, keep_lists=True, max_split_size=10, flatten_int=False):
        """Returns a flattened version of the Config as dict.

        Nested Configs and lists will be replaced by concatenated keys like so::

            {
                "a": 1,
                "b": [2, 3],
                "c": {
                    "x": 4,
                    "y": {
                        "z": 5
                    }
                },
                "d": (6, 7)
            }

        Becomes::

            {
                "a": 1,
                "b": [2, 3], # if keep_lists is True
                "b.0": 2,
                "b.1": 3,
                "c.x": 4,
                "c.y.z": 5,
                "d": (6, 7)
            }

        We return a dict because dots are disallowed within Config keys.

        Args:
            keep_lists: Keeps list along with unpacked values
            max_split_size: List longer than this will not be unpacked
            flatten_int: Integer keys will be treated as strings

        Returns:
            dict: A flattened version of self
        """

        def flat_(obj):
            def items():
                for key, val in obj.items():
                    if isinstance(val, dict) and (isinstance(key, str) or (isinstance(key, int) and flatten_int)):
                        intermediate_dict = {}
                        for subkey, subval in flat_(val).items():
                            if isinstance(subkey, str):
                                yield str(key) + "." + subkey, subval
                            elif isinstance(subkey, int) and flatten_int:
                                yield str(key) + "." + str(subkey), subval
                            else:
                                intermediate_dict[subkey] = subval
                        if len(intermediate_dict) > 0:
                            yield str(key), intermediate_dict
                    elif isinstance(val, (list, tuple)):
                        keep_this = keep_lists or not isinstance(key, (str, int)) or (isinstance(key, int) and not flatten_int)
                        if max_split_size not in (None, False) and len(val) > max_split_size:
                            keep_this = True
                        if keep_this:
                            yield key, val
                        else:
                            for i, subval in enumerate(val):
                                yield str(key) + "." + str(i), subval
                    else:
                        yield key, val

            return dict(items())

        return flat_(self)

    def to_cmd_args_str(self):
        """Create a string representing what one would need to pass to the
        command line. Does not yet use JSON encoding!

        Returns:
            str: Command line string

        """

        c_flat = self.flat()

        str_list = []
        for key, val in c_flat.items():

            if isinstance(val, (list, tuple)):
                vals = [str(v) for v in val]
                val_str = " ".join(vals)
            else:
                val_str = str(val)
            str_list.append("--{} {}".format(key, val_str))

        return "  ".join(str_list)


def update_from_sys_argv(config, warn=False):
    """Updates Config with the arguments passed as args when running the
    program. Keys will be converted to command line options, then matching
    options in `sys.argv` will be used to update the Config.

    Args:
        config (Config): Update this Config.
        warn (bool): Raise warnings if there are unknown options. Turn this on
            if you don't use any :class:`argparse.ArgumentParser` after to
            check for possible errors.

    """

    import sys
    import argparse
    import warnings

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    if len(sys.argv) > 1:

        parser = argparse.ArgumentParser(allow_abbrev=False)
        encoder = ModuleMultiTypeEncoder()
        decoder = ModuleMultiTypeDecoder()

        config_flat = config.flat()
        for key, val in config_flat.items():
            name = "--{}".format(key)
            if val is None:
                parser.add_argument(name)
            else:
                if type(val) == bool:
                    parser.add_argument(name, type=str2bool, default=val)
                elif isinstance(val, (list, tuple)):
                    if len(val) > 0 and type(val[0]) != type:
                        parser.add_argument(name, nargs='+', type=type(val[0]), default=val)
                    else:
                        parser.add_argument(name, nargs='+', default=val)
                else:
                    if type(val) == type:
                        val = encoder._encode(val)
                    parser.add_argument(name, type=type(val), default=val)

        # parse args
        param, unknown = parser.parse_known_args()
        param = vars(param)

        if len(unknown) > 0 and warn:
            warnings.warn("Called with unknown arguments: {}".format(unknown), RuntimeWarning)

        # calc diff between configs
        diff_keys = list(Config.difference_config_static(param, config_flat).flat().keys())

        # convert type args
        ignore_ = []
        for key, val in param.items():
            if val in ("none", "None"):
                param[key] = None
            if type(config_flat[key]) == type:
                if isinstance(val, str):
                    val = val.replace("\'", "")
                    val = val.replace("\"", "")
                param[key] = decoder._decode(val)
            try:
                key_split = key.split(".")
                list_object, _ = ".".join(key_split[:-1]), int(key_split[-1])
                if "--" + list_object in sys.argv:
                    ignore_.append(key)
            except ValueError as ve:
                pass
        for i in ignore_:
            del param[i]

        ### Delete not changed entries
        param_keys = list(param.keys())
        for i in param_keys:
            if i not in diff_keys and i in param:
                del param[i]

        # update dict
        config.update(param)

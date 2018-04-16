#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from vislogger.util import ModuleMultiTypeEncoder, ModuleMultiTypeDecoder


class Config(dict):

    def __init__(self, file_=None, config=None, update_from_argv=False, **kwargs):

        super(Config, self).__init__(**kwargs)
        self.__dict__ = self

        if file_ is not None:
            self.load(file_)

        if config is not None:
            self.update(config)

        self.update(kwargs)
        if update_from_argv:
            update_from_sys_argv(self)

    def dump(self, file_, indent=4, separators=(",", ": "), **kwargs):

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

    def load(self, file_, raise_=True, **kwargs):

        if hasattr(file_, "read"):
            new_dict = json.load(file_, cls=ModuleMultiTypeDecoder, **kwargs)
        else:
            with open(file_, "r") as file_object:
                new_dict = json.load(file_object, cls=ModuleMultiTypeDecoder, **kwargs)

        self.update(new_dict)

    def update_missing(self, dict_):
        for key, val in dict_.items():
            if key not in self:
                self[key] = val

    @staticmethod
    def init_objects(config):

        def init_sub_objects(objs):
            if isinstance(objs, dict):
                ret_dict = {}
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
                    elif isinstance(val, Config):
                        ret_dict[key] = Config(config=init_sub_objects(val))
                    elif isinstance(val, (dict, list, tuple, set)):
                        ret_dict[key] = init_sub_objects(val)
                    else:
                        ret_dict[key] = val
                return ret_dict
            elif isinstance(objs, (list, tuple, set)):
                ret_list = []
                for el in objs:
                    ret_list.append(init_sub_objects(el))
                return ret_list
            else:
                return objs

        conv_config = init_sub_objects(config)

        return Config(config=conv_config)

    def difference_dict(self, *other_configs):
        return self.difference_dict_static(self, *other_configs)

    @staticmethod
    def difference_dict_static(*configs):
        """Make a dict of all elements that differ between N configs.

        The resulting dict looks like this:

            {key: (config1[key], config2[key], ...)}

        If the key is missing, None will be inserted. The inputs will not be
        modified.

        Args:
            configs (Config): First config

        Returns:
            dict: Possibly empty
        """

        difference_dict = {}
        all_keys = set()
        for config in configs:
            all_keys.update(set(config.keys()))

        for key in all_keys:

            current_values = []
            all_equal = True
            for config in configs:
                if not hasattr(config, key):
                    all_equal = False
                current_values.append(getattr(config, key, None))
                if len(current_values) >= 2:
                    if current_values[-1] != current_values[-2]:
                        all_equal = False

            if not all_equal:
                difference_dict[key] = tuple(current_values)

        return difference_dict


def update_from_sys_argv(config):

    import sys
    import argparse
    import warnings

    """Updates the current config with the arguments passed as args when running the programm and removes given
    classes / converts them to None"""

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def get_key_strings(config):
        """Converts hierarichal keys into a list of keys (depth first transversal)"""

        def parse_key_strings(config_dict, prefix_key=None):
            if prefix_key is None:
                prefix_key = []

            output_list = []
            for key in config_dict.keys():
                if isinstance(config_dict[key], dict):
                    sub_tree_list = parse_key_strings(config_dict[key], prefix_key=prefix_key + [key])
                    output_list += sub_tree_list
                else:
                    output_list.append(prefix_key + [key])
            return output_list

        keys = parse_key_strings(config)
        return keys

    def get_values_for_keys(config, key_list):
        """For a list of (hierarichal) keys return the corresponding value (--> tree search)"""
        output_val = config
        for key in key_list:
            output_val = output_val.get(key, {})
        return output_val

    def set_value_for_key(config, key_list, value):
        """For a list of (hierarichal) keys set a given value (--> tree search)"""
        assert len(key_list) > 0
        for key in key_list[:-1]:
            config = config[key]
        config[key_list[-1]] = value

    def update_keys(config, update_obj):
        for update_key, update_val in update_obj.items():
            keys = update_key.split(".")
            set_value_for_key(config, keys, update_val)

    if len(sys.argv) > 1:

        parser = argparse.ArgumentParser()

        # parse just config keys
        keys = get_key_strings(config)
        for key in keys:
            val = get_values_for_keys(config, key)
            param_name = ".".join(key)
            name_str = "--%s" % param_name
            if val is None:
                parser.add_argument(name_str)
            else:
                if type(val) == bool:
                    parser.add_argument(name_str, type=str2bool, default=val)
                else:
                    parser.add_argument(name_str, type=type(val), default=val)

        # parse args
        param, unknown = parser.parse_known_args()

        if len(unknown) > 0:
            warnings.warn("Called with unknown arguments: %s" % unknown, RuntimeWarning)

        # update dict
        update_keys(config, vars(param))

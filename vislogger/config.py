#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import json
import re

class Config(dict):

    def __init__(self, file_=None, **kwargs):

        super(Config, self).__init__(**kwargs)
        self.__dict__ = self

        if file_ is not None:
            self.load(file_)

        self.update(kwargs)

    def dump(self, file_, *args, **kwargs):

        dump_data = self.copy()

        for key, val in dump_data.items():
            if type(val) == type:
                try:
                    module_ = val.__module__
                    name_ = val.__name__
                    repr_ = "type({}.{})".format(module_, name_)
                    dump_data[key] = repr_
                except Exception as e:
                    raise e

        if hasattr(file_, "write"):
            json.dump(dump_data, file_, *args, **kwargs)
        else:
            with open(file_, "w") as file_object:
                json.dump(dump_data, file_object, *args, **kwargs)

    def load(self, file_, *args, **kwargs):

        if hasattr(file_, "read"):
            new_dict = json.load(file_, *args, **kwargs)
        else:
            with open(file_, "r") as file_object:
                new_dict = json.load(file_object, *args, **kwargs)

        for key, val in new_dict.items():
            if isinstance(val, str):
                print("Check1")
                if re.match("type\(.+\)", val):
                    print("Check2")
                    try:
                        str_ = val[5:-1]
                        module_ = ".".join(str_.split(".")[:-1])
                        name_ = str_.split(".")[-1]
                        print("Check3", module_, name_)
                        type_ = getattr(importlib.import_module(module_), name_)
                        new_dict[key] = type_
                    except Exception as e:
                        raise e

        self.update(new_dict)
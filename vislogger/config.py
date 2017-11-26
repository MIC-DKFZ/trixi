#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

class Config(dict):

    def __init__(self, file_=None, **kwargs):

        super(Config, self).__init__(**kwargs)
        self.__dict__ = self

        if file_ is not None:
            self.load(file_)

        self.update(kwargs)

    def dump(self, file_, *args, **kwargs):

        if hasattr(file_, "write"):
            json.dump(self, file_, *args, **kwargs)
        else:
            with open(file_, "w") as file_object:
                json.dump(self, file_object, *args, **kwargs)

    def load(self, file_, *args, **kwargs):

        if hasattr(file_, "read"):
            new_dict = json.load(file_, *args, **kwargs)
        else:
            with open(file_, "r") as file_object:
                new_dict = json.load(file_object, *args, **kwargs)

        self.update(new_dict)
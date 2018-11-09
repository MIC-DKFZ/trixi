import json
from trixi.util import ModuleMultiTypeDecoder


class GridSearch(dict):

    def all_combinations(self):

        if len(self) == 0:
            return []

        combinations = []
        zero_key = sorted(self.keys())[0]
        inner_dict = self.copy()
        del inner_dict[zero_key]

        for val in self[zero_key]:

            if len(inner_dict) == 0:
                combinations.append({zero_key: val})
            else:
                for combination in GridSearch(inner_dict):
                    combination[zero_key] = val
                    combinations.append(combination)

        return combinations

    def __iter__(self):

        self._counter = 0
        self._all_combinations = self.all_combinations()
        self._max = len(self._all_combinations)
        return self

    def __next__(self):

        if self._counter >= self._max:
            raise StopIteration

        self._counter += 1
        return self._all_combinations[self._counter - 1]

    def read(self, file_, raise_=True, decoder_cls_=ModuleMultiTypeDecoder, **kwargs):

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

        return self

import random


class SelfShuffleList(object):
    def __init__(self, li):
        assert (type(li) is list and list)
        self.li = li  # be careful! not a deep copy!
        self.idx = 0
        self._shuffle()  # shuffle at the beginning

    def get_next(self):
        if self.idx < len(self.li):
            rtn = self.li[self.idx]
            self.idx += 1
            return rtn
        else:
            self.idx = 0
            self._shuffle()
            return self.li[self.idx]

    def _shuffle(self):
        random.Random(123).shuffle(self.li)

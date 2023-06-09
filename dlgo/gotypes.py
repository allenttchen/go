import enum
from collections import namedtuple


class Player(enum.Enum):
    black = 1
    white = 2

    @property
    def other(self):
        if self == Player.white:
            return Player.black
        else:
            return Player.white


Point_ = namedtuple('Point', ['row', 'col'])


class Point(Point_):
    def neighbors(self):
        return [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1),
        ]

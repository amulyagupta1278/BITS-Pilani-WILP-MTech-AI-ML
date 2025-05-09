from enum import IntEnum, auto


class Role(IntEnum):
    NONE = 0
    FIRE = auto()
    BUSH = auto()
    ENTRANCE = auto()
    EXIT = auto()
    EXTERIOR = auto()
    REWARD = auto()
    WALL = auto()

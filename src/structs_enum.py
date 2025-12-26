from enum import Enum

class LightState(Enum):
    RED = 0
    RED_YELLOW = 1
    GREEN = 2
    GREEN_BLINK = 3
    YELLOW = 4


class Direction(Enum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class Turn(Enum):
    STRAIGHT = 0  # Forward or Right
    LEFT = 1  # Dedicated Left Turn

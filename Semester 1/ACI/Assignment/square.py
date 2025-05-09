from dataclasses import dataclass

from border import Border
from role import Role


@dataclass(frozen=True)
class Square:
    index: int
    row: int
    column: int
    border: Border
    role: Role = Role.NONE

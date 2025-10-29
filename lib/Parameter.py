from dataclasses import dataclass

@dataclass (frozen=True)
class RBFPARAM:
    s: int
    A: list
    c: list
    w: list
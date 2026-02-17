from dataclasses import dataclass

@dataclass
class DirectionsValue:
    north: int | float
    east: int | float
    south: int | float
    west: int | float
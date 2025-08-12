from enum import Enum
from dataclasses import dataclass

# ----------------------------
# Sets: Nodes and Arcs
# ----------------------------

class ArcType (Enum):
    """
    Enumeration for different types of arcs in the network, including:
    - SERVICE: Represents service arcs where vehicles are actively serving passengers.
    - RELOCATION: Represents arcs used for relocating vehicles between zones.
    - CHARGE: Represents arcs used for charging vehicles.
    - IDLE: Represents arcs where vehicles are parked or idle.
    - WRAP: Represents wrap-around arcs, used to connect the end of the day to the start of next day.
    """
    SERVICE     = 's'  # service arc
    RELOCATION  = 'r'  # relocation arc
    CHARGE      = 'c'  # charging arc
    IDLE        = 'p'  # parking/idle arc
    WRAP        = 'w'  # wrap-around arc (end of day to start)

# dataclass decorator auto generate special method eg__init__
# frozen=True makes instances (attributes) immutable, can be used as keys in dicts or elements in sets

@dataclass(frozen=True) 
class Node:
    """
    Represents a state of a vehicle, specified by zone, time, and state of charge (SoC) level.
    """
    i: int   # zone
    t: int   # time
    l: int   # SoC level

@dataclass(frozen=True)
class Arc:
    """
    Represents a flow from one vehicle state to another.
    """
    id:     int
    type:   ArcType     # type of arc (service, relocation, etc.)
    o:      Node        # origin node
    d:      Node        # destination node
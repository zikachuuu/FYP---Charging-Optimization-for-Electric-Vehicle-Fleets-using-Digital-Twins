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


@dataclass(frozen=True)
class ServiceArc(Arc):
    """
    Represents a service arc where vehicles are actively serving passengers.
    """
    revenue:    float       # revenue earned from serving a passenger
    penalty:    float       # penalty cost for unserved demand

    def __post_init__(self):
        if self.type != ArcType.SERVICE:
            raise ValueError("ServiceArc must have type SERVICE")
        
@dataclass(frozen=True)
class ChargingArc(Arc):
    """
    Represents an arc used for charging vehicles.
    """
    charge_speed:   int     # speed of charging in SoC levels per time step

    def __post_init__(self):
        if self.type != ArcType.CHARGE:
            raise ValueError("ChargingArc must have type CHARGE")

@dataclass(frozen=True)
class RelocationArc(Arc):
    """
    Represents an arc used for relocating vehicles between zones.
    """
    def __post_init__(self):
        if self.type != ArcType.RELOCATION:
            raise ValueError("RelocationArc must have type RELOCATION")
    
@dataclass(frozen=True)
class IdleArc(Arc):
    """
    Represents an arc where vehicles are parked or idle.
    """
    def __post_init__(self):
        if self.type != ArcType.IDLE:
            raise ValueError("IdleArc must have type IDLE")
        
@dataclass(frozen=True)
class WraparoundArc(Arc):
    """
    Represents a wrap-around arc, used to connect the end of the day to the start of next day.
    """
    def __post_init__(self):
        if self.type != ArcType.WRAP:
            raise ValueError("WraparoundArc must have type WRAP")
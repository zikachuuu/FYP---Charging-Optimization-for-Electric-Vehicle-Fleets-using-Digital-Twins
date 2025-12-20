import os

from logger import Logger
from networkClass import Node, Arc, ArcType

def leader_model(
        **kwargs
    ) -> dict:
    """
    Leader: Charging Operator

    Given EV charging scheduling from the follower, calculate the electrity consumption at each time step,
    as well the variance of the electricity consumption over the day.

    Due to difficulties in calcuating penalty (it requires knowledge of previous iterations' penalties),
    we only return the variance of electricity consumption as the leader's objective.

    Penalty as well as rules to update a_t, b_t, r_t will be handled by bilevel.py
    """
    # ----------------------------
    # Parameters
    # ----------------------------
    
    # Follower parameters
    N                       : int                                       = kwargs.get("N")                       # number of operation zones (1, ..., N)
    T                       : int                                       = kwargs.get("T")                       # termination time of daily operations (0, ..., T)
    L                       : int                                       = kwargs.get("L")                       # max SoC level (all EVs start at this level) (0, ..., L)
    W                       : int                                       = kwargs.get("W")                       # maximum time intervals a passenger will wait for a ride (0, ..., W-1; demand expires at W)
    travel_demand           : dict[tuple[int, int, int], int]           = kwargs.get("travel_demand")           # travel demand from zone i to j at starting at time t
    travel_time             : dict[tuple[int, int, int], int]           = kwargs.get("travel_time")             # travel time from i to j at starting at time t
    travel_energy           : dict[tuple[int, int], int]                = kwargs.get("travel_energy")           # energy consumed for trip from zone i to j
    order_revenue           : dict[tuple[int, int, int], float]         = kwargs.get("order_revenue")           # order revenue for each trip served from i to j at time t
    penalty                 : dict[tuple[int, int, int], float]         = kwargs.get("penalty")                 # penalty cost for each unserved trip from i to j at time t 
    L_min                   : int                                       = kwargs.get("L_min")                   # min SoC level all EV must end with at the end of the daily operations
    num_EVs                 : int                                       = kwargs.get("num_EVs")                 # total number of EVs in the fleet
    num_ports               : dict[int, int]                            = kwargs.get("num_ports")               # number of chargers in each zone
    elec_supplied           : dict[tuple[int, int], int]                = kwargs.get("elec_supplied")           # electricity supplied (in SoC levels) at zone i at time t
    max_charge_speed        : int                                       = kwargs.get("max_charge_speed")        # max charging speed (in SoC levels) of one EV in one time step
        
    charge_cost_low         : dict[int, float]                          = kwargs.get("charge_cost_low")         # charge cost per unit of SoC at zone i at time t when usage is below threshold
    charge_cost_high        : dict[int, float]                          = kwargs.get("charge_cost_high")        # charge cost per unit of SOC at zone i at time t when usage is above threshold
    elec_threshold          : dict[int, int]                            = kwargs.get("elec_threshold")          # electricity threshold at zone i at time t

    # leader parameters
    penalty_weight          : float                                     = kwargs.get("penalty_weight")          # weight for penalty on high price settings
    wholesale_elec_price    : dict[int, float]                          = kwargs.get("wholesale_elec_price")    # wholesale electricity price at time t ($ per kWh)
    
    # Arcs
    all_arcs                : dict[int                  , Arc]          = kwargs["all_arcs"]
    type_arcs               : dict[ArcType              , set[int]]     = kwargs["type_arcs"]
    in_arcs                 : dict[Node                 , set[int]]     = kwargs["in_arcs"]
    out_arcs                : dict[Node                 , set[int]]     = kwargs["out_arcs"]
    service_arcs_ijt        : dict[tuple[int, int, int] , set[int]]     = kwargs["service_arcs_ijt"]
    charge_arcs_it          : dict[tuple[int, int]      , set[int]]     = kwargs["charge_arcs_it"]
    charge_arcs_t           : dict[int                  , set[int]]     = kwargs["charge_arcs_t"]

    # Sets
    valid_travel_demand     : dict[tuple[int, int, int], int]           = kwargs["valid_travel_demand"]
    invalid_travel_demand   : set[tuple[int, int, int]]                 = kwargs["invalid_travel_demand"]
    ZONES                   : list[int]                                 = kwargs["ZONES"]
    TIMESTEPS               : list[int]                                 = kwargs["TIMESTEPS"]
    LEVELS                  : list[int]                                 = kwargs["LEVELS"]
    AGES                    : list[int]                                 = kwargs["AGES"]

    # Metadata
    to_console              : bool                                      = kwargs.get("to_console", False)       # whether to log to console
    to_file                 : bool                                      = kwargs.get("to_file", True)           # whether to log to file
    timestamp               : str                                       = kwargs.get("timestamp", "")           # timestamp for logging
    file_name               : str                                       = kwargs.get("file_name", "")           # filename for logging
    folder_name             : str                                       = kwargs.get("folder_name", "")         # folder name for logs and results


    # Create logger
    logger = Logger("model_leader", level="DEBUG", to_console=to_console, timestamp=timestamp)
    if to_file:
        logger.save (os.path.join (folder_name, f"model_leader_{file_name}"))

    logger.info("Parameters loaded successfully into leader model.")


    # ----------------------------
    # Leader model computations
    # ----------------------------

    # calulate electricity consumption at each time step
    electricity_usage: list[float] = [0] * (T + 1)

    for t in TIMESTEPS:
        if t == 0 or t == T:
            continue  # No charging at time 0 or T

        # Calculate total electricity used at time t
        for e_id in charge_arcs_t.get(t, set()):
            arc = all_arcs[e_id]
            # Electricity used = number of EVs * charge amount
            charge_amount = arc.d.l - arc.o.l  # SoC levels charged
            electricity_usage[t] += x[e_id] * charge_amount


    # calculate variance of electricity consumption
    mean_usage = sum(electricity_usage[1:T]) / (T - 1)  # exclude time 0 and T
    variance = sum((usage - mean_usage)**2 for usage in electricity_usage[1:T]) / (T - 1)

    logger.info("Leader model computation completed.")
    return variance


        
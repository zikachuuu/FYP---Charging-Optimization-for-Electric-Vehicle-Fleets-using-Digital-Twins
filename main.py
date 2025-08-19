import gurobipy as gp

from logger import Logger
from model import model
from networkClass import Node, Arc, ArcType

logger = Logger("main", level="DEBUG", to_console=True)
logger.save("main")

if __name__ == "__main__":

    file_name = input ("Enter the JSON test case file name, without the .json extension. It should be located in the Testcases folder: ").strip()

    # Load data from the specified JSON file
    data: dict = None
    try:
        with open(f"Testcases/{file_name}.json", "r") as f:
            logger.info(f"Loading data from {file_name}.json")

            import json
            data = json.load(f)

            logger.info("Data loaded successfully.")

    except FileNotFoundError as e:
        logger.error(f"File {file_name}.json not found. Please ensure it exists in the Testcases folder.")
        exit(1)

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_name}.json: {e}")
        exit(1)
    
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data: {e}")
        exit(1)

    N               : int                               = data.get("N")               # number of operation zones
    T               : int                               = data.get("T")               # termination time of daily operations
    L               : int                               = data.get("L")               # max SoC level
    travel_demand   : dict[tuple[int, int, int], int]   = data.get("travel_demand")   # d_ijt = travel demand from zone i to j at time t
    travel_time     : dict[tuple[int, int, int], int]   = data.get("travel_time")     # τ_ijt = travel time from i to j at time t
    travel_energy   : dict[tuple[int, int], int]        = data.get("travel_energy")   # l_ij = energy consumed for trip i->j
    order_revenue   : dict[tuple[int, int, int], float] = data.get("order_revenue")   # p_e = order revenue for service arcs departing (i,t) to j
    penalty         : dict[tuple[int, int, int], float] = data.get("penalty")         # c_ijt = penalty cost for unserved demand
    charge_speed    : int                               = data.get("charge_speed")    # charge speed (SoC levels per timestep)
    num_ports       : dict[int, int]                    = data.get("num_ports")       # number of chargers in each zone
    num_EVs         : int                               = data.get("num_EVs")         # total number of EVs in the fleet
    charge_cost     : dict[int, float]                  = data.get("charge_cost")     # c^c_e = cost for charging arcs

    output = model(
        N               = N,
        T               = T,
        L               = L,
        travel_demand   = travel_demand,
        travel_time     = travel_time,
        travel_energy   = travel_energy,
        order_revenue   = order_revenue,
        penalty         = penalty,
        charge_speed    = charge_speed,
        num_ports       = num_ports,
        num_EVs         = num_EVs,
        charge_cost     = charge_cost
    )
    logger.info("Solution retreived from model.")

    # Extract variables and sets from the output
    obj                     : float                                     = output["obj"]
    x                       : dict[int, int]                            = output["sol"]["x"]  # flow variables
    s                       : dict[tuple[int, int, int], int]           = output["sol"]["s"]  # unserved demand variables
    total_service_revenue   : float                                     = output["sol"]["total_service_revenue"]
    total_penalty_cost      : float                                     = output["sol"]["total_penalty_cost"]
    total_charge_cost       : float                                     = output["sol"]["total_charge_cost"]

    all_arcs            : dict[int                  , Arc]              = output["sets"]["all_arcs"]
    type_arcs           : dict[ArcType              , set[int]]         = output["sets"]["type_arcs"]
    in_arcs             : dict[Node                 , set[int]]         = output["sets"]["in_arcs"]
    out_arcs            : dict[Node                 , set[int]]         = output["sets"]["out_arcs"]
    type_starting_arcs_l: dict[tuple[ArcType, int]  , set[int]]         = output["sets"]["type_starting_arcs_l"]
    type_ending_arcs_l  : dict[tuple[ArcType, int]  , set[int]]         = output["sets"]["type_ending_arcs_l"]
    service_arcs_ijt    : dict[tuple[int, int, int] , set[int]]         = output["sets"]["service_arcs_ijt"]
    charge_arcs_it      : dict[tuple[int, int]      , set[int]]         = output["sets"]["charge_arcs_it"]

    for arc_id, arc in all_arcs.items():
        if x.get(arc_id, 0) <= 0:
            continue
        logger.info(f"Arc {arc_id} ({arc.type.name}): ")
        logger.info(f"  From ({arc.o.i}, {arc.o.t}, {arc.o.l}) to ({arc.d.i}, {arc.d.t}, {arc.d.l})")
        logger.info(f"  Flow: {x.get(arc_id, 0)}")
        logger.info(f"  Unserved Demand: {s.get((arc.o.i, arc.d.i, arc.o.t), 0) if arc.type == ArcType.SERVICE else 'N/A'}")
    
    total_trips_served = sum(x[arc_id] for arc_id in type_arcs[ArcType.SERVICE])
    total_trips_served_alt = sum(s[(arc.o.i, arc.d.i, arc.o.t)]
                                for arc_id in type_arcs[ArcType.SERVICE]
                                for arc in [all_arcs[arc_id]]
                                if arc.type == ArcType.SERVICE)
    
    if total_trips_served != total_trips_served_alt:
        logger.error("Total trips served (direct calculation) does not match alternative calculation. Check your model.")
        logger.debug(f"Direct calculation: {total_trips_served}, Alternative calculation: {total_trips_served_alt}")

    logger.info ("Vehicle Operations Summary:")
    logger.info (f"  Total trips served: {total_trips_served:.2f}")
    logger.info (f"  Trips per vehicle: {total_trips_served / num_EVs:.2f}")


        


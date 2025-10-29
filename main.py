import gurobipy as gp
from datetime import datetime
import json
import math
import os

from logger import Logger
from model import model
from networkClass import Node, Arc, ArcType
from utility import convert_key_types
from postprocessing import postprocessing

timestamp = datetime.now().strftime("%Y%m%d_%H%M")

logger = Logger("main", level="DEBUG", to_console=False, timestamp=timestamp)

if __name__ == "__main__":
    print ("---------------------------------------------------")
    print ("Welcome to the EV Fleet Charging Optimization Model")
    print ("---------------------------------------------------")    
    print ()
    print ("Find input test case files in the Testcases folder.")
    print ("Output results will be saved in the Results folder.")
    print ("Log files for debugging will be saved in the Logs folder.")
    print ()
    directory = input ("Enter the specific folder in the Testcases folder (or press Enter if the file is directly in the Testcases folder): ").strip()
    file_name = input ("Enter the JSON test case file name, without the .json extension. It should be located in the specified folder: ").strip()
    print ()
    print ("Give a unique name for this run, to be used as folder name for the log files and results.")
    folder_name = input ("If empty, default name of <testcase file name>_<timestamp> will be used: ").strip()
    print()

    if folder_name == "":
        folder_name = f"{file_name}_{timestamp}"
    else:
        folder_name = f"{folder_name}_{file_name}_{timestamp}"

    # Create necessary directories if they do not exist
    if not os.path.exists("Logs"):
        os.makedirs("Logs")
    if not os.path.exists(os.path.join("Logs", folder_name)):
        os.makedirs(os.path.join("Logs", folder_name))
    if not os.path.exists("Results"):
        os.makedirs("Results")
    if not os.path.exists(os.path.join("Results", folder_name)):
        os.makedirs(os.path.join("Results", folder_name))

    logger.save(os.path.join(folder_name, f"main_{file_name}"))  # Save log file in the specific folder for this run

    results_name = os.path.join("Results", folder_name, f"results_{file_name}_{timestamp}.xlsx")

    # Load data from the specified JSON file
    processed_data: dict = None
    try:
        with open(os.path.join ("Testcases", directory, file_name + ".json"), "r") as f:
            logger.info(f"Loading data from {file_name}.json")

            data = json.load(f)
            # logger.debug(f"Raw data loaded: {data}")
            logger.info ("Converting keys in data to appropriate types...")

            processed_data = convert_key_types(data)  # Convert keys to tuples of integers
            # logger.debug(f"Processed data: {processed_data}")

            logger.info("Data loaded successfully.")

    except FileNotFoundError as e:
        logger.error(f"File {file_name}.json not found. Please ensure it exists in the Testcases folder with json extension.")
        exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_name}.json: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data: {e}")
        exit(1)

    # Extract parameters from processed_data
    try:
        N               : int                               = processed_data["N"]               # number of operation zones (1, ..., N)
        T               : int                               = processed_data["T"]               # termination time of daily operations (0, ..., T)
        L               : int                               = processed_data["L"]               # max SoC level (all EVs start at this level) (0, ..., L)
        W               : int                               = processed_data["W"]               # maximum time intervals a passenger will wait for a ride (0, ..., W-1; demand expires at W)
        
        travel_demand   : dict[tuple[int, int, int], int]   = processed_data["travel_demand"]   # travel demand from zone i to j at starting at time t
        travel_time     : dict[tuple[int, int, int], int]   = processed_data["travel_time"]     # travel time from i to j at starting at time t
        travel_energy   : dict[tuple[int, int], int]        = processed_data["travel_energy"]   # energy consumed for trip from zone i to j
        order_revenue   : dict[tuple[int, int, int], float] = processed_data["order_revenue"]   # order revenue for each trip served from i to j at time t
        penalty         : dict[tuple[int, int, int], float] = processed_data["penalty"]         # penalty cost for each unserved trip from i to j at time t
        L_min           : int                               = processed_data["L_min"]           # min SoC level all EV must end with at the end of the daily operations
        num_EVs         : int                               = processed_data["num_EVs"]         # total number of EVs in the fleet
        
        num_ports           : dict[int, int]                = processed_data["num_ports"]               # number of charging ports in each zone
        elec_supplied       : dict[tuple[int, int], int]    = processed_data["elec_supplied"]           # electricity supplied (in SoC levels) at zone i at time t
        max_charge_speed    : int                           = processed_data["max_charge_speed"]        # max charging speed (in SoC levels) of one EV in one time step
        wholesale_elec_price: dict[int, float]              = processed_data["wholesale_elec_price"]    # wholesale electricity price at time t
        
        charge_cost_low : dict[int, float]                  = processed_data["charge_cost_low"]         # charge cost per unit of SoC at zone i at time t when usage is below threshold
        charge_cost_high: dict[int, float]                  = processed_data["charge_cost_high"]        # charge cost per unit of SOC at zone i at time t when usage is above threshold
        elec_threshold  : dict[int, int]                    = processed_data["elec_threshold"]          # electricity threshold at zone i at time t

    except KeyError as e:
        logger.error(f"Missing required parameter in input data: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while extracting parameters: {e}")
        exit(1)
    
    output = model(
        N                       = N                     ,
        T                       = T                     ,
        L                       = L                     ,
        W                       = W                     ,

        travel_demand           = travel_demand         ,
        travel_time             = travel_time           ,
        travel_energy           = travel_energy         ,
        order_revenue           = order_revenue         ,
        penalty                 = penalty               ,
        L_min                   = L_min                 ,
        num_EVs                 = num_EVs               ,

        num_ports               = num_ports             ,
        elec_supplied           = elec_supplied         ,
        max_charge_speed        = max_charge_speed      ,
        wholesale_elec_price    = wholesale_elec_price  ,

        charge_cost_low         = charge_cost_low       ,
        charge_cost_high        = charge_cost_high      ,
        elec_threshold          = elec_threshold        ,

        # Metadata
        timestamp               = timestamp             ,
        file_name               = file_name             ,
        folder_name             = folder_name           ,
    )
    logger.info("Solution retreived from model.")

    if output == None:
        logger.error ("Optimization failed...")
        exit (1)

    # Extract variables and sets from the output
    obj                     : float                                     = output["obj"]
    
    x                       : dict[int, int]                            = output["sol"]["x"]  
    s                       : dict[tuple[int, int, int], int]           = output["sol"]["s"]  
    u                       : dict[tuple[int, int, int, int], int]      = output["sol"]["u"]
    e                       : dict[tuple[int, int, int], int]           = output["sol"]["e"]
    z                       : dict[int, int]                            = output["sol"]["z"]
    y                       : dict[int, float]                          = output["sol"]["y"]
    h                       : float                                     = output["sol"]["h"]  
    l                       : float                                     = output["sol"]["l"]
    service_revenues        : dict[int, float]                          = output["sol"]["service_revenues"]
    penalty_costs           : dict[int, float]                          = output["sol"]["penalty_costs"]
    charge_costs            : dict[int, float]                          = output["sol"]["charge_costs"]                                                                                        

    all_arcs                : dict[int                  , Arc]          = output["arcs"]["all_arcs"]
    type_arcs               : dict[ArcType              , set[int]]     = output["arcs"]["type_arcs"]
    in_arcs                 : dict[Node                 , set[int]]     = output["arcs"]["in_arcs"]
    out_arcs                : dict[Node                 , set[int]]     = output["arcs"]["out_arcs"]
    service_arcs_ijt        : dict[tuple[int, int, int] , set[int]]     = output["arcs"]["service_arcs_ijt"]
    charge_arcs_it          : dict[tuple[int, int]      , set[int]]     = output["arcs"]["charge_arcs_it"]
    charge_arcs_t           : dict[int                  , set[int]]     = output["arcs"]["charge_arcs_t"]

    valid_travel_demand     : dict[tuple[int, int, int], int]           = output["sets"]["valid_travel_demand"]
    invalid_travel_demand   : set[tuple[int, int, int]]                 = output["sets"]["invalid_travel_demand"]
    ZONES                   : list[int]                                 = output["sets"]["ZONES"]
    TIMESTEPS               : list[int]                                 = output["sets"]["TIMESTEPS"]
    LEVELS                  : list[int]                                 = output["sets"]["LEVELS"]
    AGES                    : list[int]                                 = output["sets"]["AGES"]


    # ----------------------------
    # Arcs Information
    # ----------------------------    
    logger.debug("Arcs information:")
    for id, arc in all_arcs.items():
        logger.debug (f"  Arc {id}: type {arc.type} from node ({arc.o.i}, {arc.o.t}, {arc.o.l}) to ({arc.d.i}, {arc.d.t}, {arc.d.l}); flow: {x[id]}")
    
    
    # ----------------------------
    # Summarised Information
    # ----------------------------
    total_service_revenue   : float = sum(service_revenues.values())
    total_penalty_cost      : float = sum(penalty_costs.values())
    total_charge_cost       : float = sum(charge_costs.values())

    logger.info("Model results:")
    logger.info(f"  Objective value: {obj:.2f}")
    logger.info(f"  Total service revenue: {total_service_revenue:.2f}")
    logger.info(f"  Total penalty cost: {total_penalty_cost:.2f}")
    logger.info(f"  Total charge cost: {total_charge_cost:.2f}")    
    
    # ----------------------------
    # EV flow in each arcs
    # ----------------------------
    logger.info("EV flow in arcs:")
    curr_time = -1
    for arc_id, arc in sorted(
        all_arcs.items(),
        key = lambda item: (item[1].o.t, item[1].o.i, item[1].d.i)
    ):
        if x.get(arc_id, 0) <= 0:
            # we skip this arc if the arc has no flow and no unserved demand
            continue

        if curr_time != arc.o.t:
            curr_time = arc.o.t
            logger.info (f"  Time {curr_time}:")

        logger.info(f"    Arc {arc_id} ({arc.type.name}): ")
        logger.info(f"      From ({arc.o.i}, {arc.o.t}, {arc.o.l}) to ({arc.d.i}, {arc.d.t}, {arc.d.l})")
        logger.info(f"      Flow: {x.get(arc_id, 0)}")
        if arc.type == ArcType.SERVICE:
            logger.info(f"      Unserved Demand: {s.get((arc.o.i, arc.d.i, arc.o.t), 0)}")

    # ----------------------------
    # Calculated information
    # ----------------------------

    # total number of valid trips
    total_trips: int = sum (
        valid_travel_demand.values()
    )
    # total number of valid trips served
    total_trips_served: int = sum(
        x.get(arc_id, 0) 
        for arc_id in type_arcs[ArcType.SERVICE]
    )
    # total number of valid trips unserved (carried over from all intervals)
    total_trips_unserved: int = sum (
        e[(i, j, t)]
        for i in ZONES
        for j in ZONES
        for t in TIMESTEPS
    ) + sum (
        s[(i, j, T)]
        for i in ZONES
        for j in ZONES
    )
    # total number of time intervals spent on service (sum of all EVs)
    total_service_time: int = sum (
        x[e] * (all_arcs[e].d.t - all_arcs[e].o.t)
        for e in type_arcs[ArcType.SERVICE]
    )
    # total number of time intervals spent on charging (sum of all EVs)
    total_charge_time: int = sum (
        x[e] * (all_arcs[e].d.t - all_arcs[e].o.t)
        for e in type_arcs[ArcType.CHARGE]    
    )
    # total number of time intervals spent on relocation (sum of all EVs)
    total_relocation_time: int = sum (
        x[e] * (all_arcs[e].d.t - all_arcs[e].o.t)
        for e in type_arcs[ArcType.RELOCATION]    
    )
    # total number of time intervals spent on idle (sum of all EVs)
    total_idle_time: int = sum (
        x[e] * (all_arcs[e].d.t - all_arcs[e].o.t)
        for e in type_arcs[ArcType.IDLE]    
    )

    if total_trips != total_trips_served + total_trips_unserved:
        logger.warning (f"  Total trips requested ({total_trips}) does not match total trips served ({total_trips_served}) + unserved ({total_trips_unserved}).")

    if total_service_time + total_charge_time + total_relocation_time + total_idle_time != T * num_EVs:
        logger.warning (f"  Total time does not sum up! Total time: {T * num_EVs}, Service time: {total_service_time}, Charge time: {total_charge_time}, Relocation time: {total_relocation_time}, Idle time: {total_idle_time}")

    postprocessing(
        # Input parameters
        N                       = N                     ,
        T                       = T                     ,
        L                       = L                     ,
        W                       = W                     ,

        travel_demand           = travel_demand         ,
        travel_time             = travel_time           ,
        travel_energy           = travel_energy         ,
        order_revenue           = order_revenue         ,
        penalty                 = penalty               ,
        L_min                   = L_min                 ,
        num_EVs                 = num_EVs               ,

        num_ports               = num_ports             ,
        elec_supplied           = elec_supplied         ,
        max_charge_speed        = max_charge_speed      ,
        wholesale_elec_price    = wholesale_elec_price  ,

        charge_cost_low         = charge_cost_low       ,
        charge_cost_high        = charge_cost_high      ,
        elec_threshold          = elec_threshold        ,

        # Metadata
        timestamp               = timestamp             ,
        file_name               = file_name             ,
        results_name            = results_name          ,
        folder_name             = folder_name           ,
        
        # Output results
        obj                     = obj                   ,
        x                       = x                     ,
        s                       = s                     ,
        u                       = u                     ,
        e                       = e                     ,
        z                       = z                     ,
        y                       = y                     ,
        h                       = h                     ,
        l                       = l                     ,
        service_revenues        = service_revenues       ,
        penalty_costs           = penalty_costs         ,
        charge_costs            = charge_costs          ,
        total_service_revenue   = total_service_revenue ,
        total_penalty_cost      = total_penalty_cost    ,
        total_charge_cost       = total_charge_cost     ,

        # Arcs
        all_arcs                = all_arcs              ,
        type_arcs               = type_arcs             ,
        in_arcs                 = in_arcs               ,
        out_arcs                = out_arcs              ,
        service_arcs_ijt        = service_arcs_ijt      ,
        charge_arcs_it          = charge_arcs_it        ,
        charge_arcs_t           = charge_arcs_t         ,

        # Sets
        valid_travel_demand     = valid_travel_demand   ,
        invalid_travel_demand   = invalid_travel_demand ,
        ZONES                   = ZONES                 ,
        TIMESTEPS               = TIMESTEPS             ,
        LEVELS                  = LEVELS                ,
        AGES                    = AGES                  ,
    )


        


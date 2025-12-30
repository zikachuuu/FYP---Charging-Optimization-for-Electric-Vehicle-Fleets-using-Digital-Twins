from datetime import datetime
import json
import os
import traceback
import math

from logger import Logger
from model_follower import follower_model, follower_model_builder
from networkClass import Node, Arc, ArcType
from utility import convert_key_types
from postprocessing import postprocessing
from bilevel_DE import run_parallel_de

POP_SIZE        : int   = 16        # population size for DE (should be in multiples of <number of CPU cores> - 1)
MAX_ITER        : int   = 2         # maximum iterations for DE
DIFF_WEIGHT     : float = 0.9       # Differential Weight / Mutation: Controls jump size
                                    #   Higher = bigger jumps (exploration); Lower = fine-tuning (exploitation)
                                    #   Since population size is small, need to aggresively explore to avoid getting stuck.
CROSS_PROB      : float = 0.9       # Crossover Probability: How much DNA comes from the mutant vs. the parent
                                    #   0.9 means 90% of the genes change every step.
                                    #   We want to mix good genes quickly, so set it high.
VAR_THRESHOLD   : float = 4         # variance threshold for early stopping of DE
PENALTY_WEIGHT  : float = 0.5       # penalty weight for high price settings in leader fitness function
NUM_ANCHORS     : int   = 4         # number of anchors for DE
VARS_PER_STEP   : int   = 3         # number of dimensions (variables) per time step (i.e. a_t, b_t, r_t)

"""
Notes for setting POP_SIZE and MAX_ITER:
    - POP_SIZE should be a multiple of number of CPU cores - 1 (1 core is used for main process)
    - number of CPU cores - 1 candidates are evaluated in parallel in each iteration
    - Time to iterate the entire population once = time to evaluate one candidate * ceil (POP_SIZE / (num_cores - 1))
    - MAX_ITER = (desired max runtime) / (time to iterate entire population once)
    - Check time to evaluate one candidate by running only the follower model (Stage 2) by choosing model_choice = '1' in the main.py
"""


if __name__ == "__main__":
    print ()
    print ("-------------------------------------------------------")
    print ("| Welcome to the EV Fleet Charging Optimization Model |")
    print ("-------------------------------------------------------")    
    print ()
    print ("    - Find input test case files in the Testcases folder.")
    print ("    - Output results will be saved in the Results folder.")
    print ("    - Log files for debugging will be saved in the Logs folder.")
    print ()
    print ("You can choose to run only the follower model (Stage 2), or run the full bilevel optimization model (Stage 1 + Stage 2).")
    print ("If you choose to run only the follower model, you need to provide the pricing and threshold values in the input JSON file.")
    print ("Specfically, provide 'charge_cost_low', 'charge_cost_high', and 'elec_threshold' values for each time step (0 to T inclusive).")
    print ()
    print ("------------------------------------------------------")
    print ()
    model_choice = input ("Enter '1' to run only the follower model, or press '2' to run the bilevel optimization model (default is '2'): \n").strip()
    print ()
    directory = input ("Enter the specific folder in the Testcases folder (or press Enter if the file is directly in the Testcases folder): \n").strip()
    print ()
    file_name = input ("Enter the JSON test case file name, without the .json extension. It should be located in the specified folder: \n").strip()
    print ()
    print ("Give a unique name for this run, to be used as folder name for the log files and results.")
    folder_name = input ("If empty, default name of <testcase file name>_<timestamp> will be used: \n").strip()
    print ()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if folder_name == "":
        folder_name = f"{file_name}_{timestamp}"
    else:
        folder_name = f"{folder_name}_{file_name}_{timestamp}"

    # Create necessary directories if they do not exist
    os.makedirs("Logs", exist_ok=True)
    os.makedirs(os.path.join("Logs", folder_name), exist_ok=True)
    os.makedirs("Results", exist_ok=True)
    os.makedirs(os.path.join("Results", folder_name), exist_ok=True)

    # Set up logger
    logger = Logger(
        "main"                      , 
        level       = "DEBUG"       , 
        to_console  = True          , 
        folder_name = folder_name   , 
        file_name   = file_name     , 
        timestamp   = timestamp     ,    
    )
    logger.save()

    results_name = os.path.join("Results", folder_name, f"results_{file_name}_{timestamp}.xlsx")

    # Load data from the specified JSON file
    processed_data: dict = None
    try:
        with open(os.path.join ("Testcases", directory, file_name + ".json"), "r") as F:
            logger.info(f"Loading data from {file_name}.json")

            data = json.load(F)
            # logger.debug(f"Raw data loaded: {data}")
            logger.info ("Converting keys in data to appropriate types...")

            processed_data = convert_key_types(data)  # Convert keys to tuples of integers
            # logger.debug(f"Processed data: {processed_data}")

            logger.info ("Data loaded successfully.")

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
        # Follower model parameters
        N                   : int                                   = processed_data["N"]                       # number of operation zones (1, ..., N)
        T                   : int                                   = processed_data["T"]                       # termination time of daily operations (0, ..., T)
        L                   : int                                   = processed_data["L"]                       # max SoC level (all EVs start at this level) (0, ..., L)
        W                   : int                                   = processed_data["W"]                       # maximum time intervals a passenger will wait for a ride (0, ..., W-1; demand expires at W)
        travel_demand       : dict[tuple[int, int, int] , int]      = processed_data["travel_demand"]           # travel demand from zone i to j at starting at time t
        travel_time         : dict[tuple[int, int, int] , int]      = processed_data["travel_time"]             # travel time from i to j at starting at time t
        travel_energy       : dict[tuple[int, int]      , int]      = processed_data["travel_energy"]           # energy consumed for trip from zone i to j
        order_revenue       : dict[tuple[int, int, int] , float]    = processed_data["order_revenue"]           # order revenue for each trip served from i to j at time t
        penalty             : dict[tuple[int, int, int] , float]    = processed_data["penalty"]                 # penalty cost for each unserved trip from i to j at time t
        L_min               : int                                   = processed_data["L_min"]                   # min SoC level all EV must end with at the end of the daily operations
        num_EVs             : int                                   = processed_data["num_EVs"]                 # total number of EVs in the fleet 
        num_ports           : dict[int                  , int]      = processed_data["num_ports"]               # number of charging ports in each zone
        elec_supplied       : dict[tuple[int, int]      , int]      = processed_data["elec_supplied"]           # electricity supplied (in SoC levels) at zone i at time t
        max_charge_speed    : int                                   = processed_data["max_charge_speed"]        # max charging speed (in SoC levels) of one EV in one time step
        
        # Leader model parameters
        wholesale_elec_price: dict[int                  , float]    = processed_data["wholesale_elec_price"]    # wholesale electricity price at time t

    except KeyError as e:
        logger.error(f"Missing required parameter in input data: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while extracting parameters: {e}")
        exit(1)
    

    # -----------------------------------------------------------
    # Build the network for the follower model (reusable)
    # -----------------------------------------------------------
    network = follower_model_builder(
        # Follower model parameters
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

        # Metadata
        timestamp               = timestamp             ,
        file_name               = file_name             ,
        folder_name             = folder_name           ,
    )

    # Extract reusable network components
    V_set                  : set[Node]                              = network["V_set"]
    
    all_arcs               : dict[int                   , Arc]      = network["all_arcs"]
    type_arcs              : dict[ArcType               , set[int]] = network["type_arcs"]
    in_arcs                : dict[Node                  , set[int]] = network["in_arcs"]
    out_arcs               : dict[Node                  , set[int]] = network["out_arcs"]
    service_arcs_ijt       : dict[tuple[int, int, int]  , set[int]] = network["service_arcs_ijt"]
    charge_arcs_it         : dict[tuple[int, int]       , set[int]] = network["charge_arcs_it"]
    charge_arcs_t          : dict[int                   , set[int]] = network["charge_arcs_t"]
    valid_travel_demand    : dict[tuple[int, int, int]  , int]      = network["valid_travel_demand"]
    invalid_travel_demand  : set[tuple[int, int, int]]              = network["invalid_travel_demand"]
    
    ZONES                  : list[int]                              = network["ZONES"]
    TIMESTEPS              : list[int]                              = network["TIMESTEPS"]
    LEVELS                 : list[int]                              = network["LEVELS"]
    AGES                   : list[int]                              = network["AGES"]


    # -----------------------------------------------------------
    # Stage 1: Run the bilevel optimization on relaxed model
    # -----------------------------------------------------------
    if model_choice == "1":
        logger.info("Stage 1 skipped as per user choice. Running only the follower model (Stage 2).")
        try:
            charge_cost_low     : dict[int, float]                  = processed_data["charge_cost_low"]       # a_t
            charge_cost_high    : dict[int, float]                  = processed_data["charge_cost_high"]      # b_t
            elec_threshold      : dict[int, int]                    = processed_data["elec_threshold"]        # r_t
        except KeyError as e:
            logger.error(f"Missing required pricing/threshold parameter in input data for follower model: {e}")
            exit(1)

    else:
        logger.info("Stage 1: Starting bilevel optimization using Differential Evolution...")
        try:
            best_solution = run_parallel_de(
                # Follower model parameters
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

                # Network components
                V_set                   = V_set                 ,
                all_arcs                = all_arcs              ,
                type_arcs               = type_arcs             ,
                in_arcs                 = in_arcs               ,
                out_arcs                = out_arcs              ,
                service_arcs_ijt        = service_arcs_ijt      ,
                charge_arcs_it          = charge_arcs_it        ,
                charge_arcs_t           = charge_arcs_t         ,
                valid_travel_demand     = valid_travel_demand   ,
                invalid_travel_demand   = invalid_travel_demand ,
                ZONES                   = ZONES                 ,
                TIMESTEPS               = TIMESTEPS             ,
                LEVELS                  = LEVELS                ,
                AGES                    = AGES                  ,

                # Leader model parameters
                wholesale_elec_price    = wholesale_elec_price  ,
                PENALTY_WEIGHT          = PENALTY_WEIGHT        ,

                # DE parameters
                POP_SIZE                = POP_SIZE              ,
                MAX_ITER                = MAX_ITER              ,
                DIFF_WEIGHT             = DIFF_WEIGHT           ,
                CROSS_PROB              = CROSS_PROB            ,
                VAR_THRESHOLD           = VAR_THRESHOLD         ,
                NUM_ANCHORS             = NUM_ANCHORS           ,
                VARS_PER_STEP           = VARS_PER_STEP         ,

                # Metadata
                timestamp               = timestamp             ,
                file_name               = file_name             ,
                folder_name             = folder_name           ,
            )
        except Exception as e:
            logger.error(f"Bilevel optimization failed in Stage 1: {e}")
            logger.error(traceback.format_exc())
            exit(1)
        
        logger.info("Bilevel optimization completed.")

        charge_cost_low     : dict[int, float]                  = best_solution["charge_cost_low"]       # a_t
        charge_cost_high    : dict[int, float]                  = best_solution["charge_cost_high"]      # b_t
        elec_threshold      : dict[int, int]                    = best_solution["elec_threshold"]        # r_t

    # ----------------------------------------------------------------------------------
    # Stage 2: Run the original follower model with the obtained pricing and threshold
    # ----------------------------------------------------------------------------------
    logger.info("Stage 2: Running follower model with obtained pricing and threshold...")

    try:
        follower_outputs = follower_model(
            # Follower model parameters
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

            # Network components
            V_set                   = V_set                 ,
            all_arcs                = all_arcs              ,
            type_arcs               = type_arcs             ,
            in_arcs                 = in_arcs               ,
            out_arcs                = out_arcs              ,
            service_arcs_ijt        = service_arcs_ijt      ,
            charge_arcs_it          = charge_arcs_it        ,
            charge_arcs_t           = charge_arcs_t         ,
            valid_travel_demand     = valid_travel_demand   ,
            invalid_travel_demand   = invalid_travel_demand ,
            ZONES                   = ZONES                 ,
            TIMESTEPS               = TIMESTEPS             ,
            LEVELS                  = LEVELS                ,
            AGES                    = AGES                  ,

            # Pricing and threshold
            charge_cost_low         = charge_cost_low       ,
            charge_cost_high        = charge_cost_high      ,
            elec_threshold          = elec_threshold        ,

            # Metadata
            relaxed                 = True                  ,
            to_console              = False                 ,
            to_file                 = True                  ,
            timestamp               = timestamp             ,
            file_name               = file_name             ,
            folder_name             = folder_name           ,
        )
    except Exception as e:
        logger.error(f"Follower model failed in Stage 2: {e}")
        logger.error(traceback.format_exc())
        exit(1)
    
    logger.info("Solution retreived from model.")

    # Extract variables and sets from the output
    obj             : float                                     = follower_outputs["obj"]
    x               : dict[int, float]                          = follower_outputs["x"]
    s               : dict[tuple[int, int, int]     , float]    = follower_outputs["s"]
    u               : dict[tuple[int, int, int, int], float]    = follower_outputs["u"]
    e               : dict[tuple[int, int, int]     , float]    = follower_outputs["e"]
    q               : dict[int                      , float]    = follower_outputs["q"]
    service_revenues: dict[int                      , float]    = follower_outputs["service_revenues"]
    penalty_costs   : dict[int                      , float]    = follower_outputs["penalty_costs"]
    charge_costs    : dict[int                      , float]    = follower_outputs["charge_costs"]

    # ----------------------------
    # Arcs Information
    # ----------------------------    
    logger_arcs = Logger(
        "arcs"                      , 
        level       = "DEBUG"       , 
        to_console  = False         , 
        folder_name = folder_name   , 
        file_name   = file_name     , 
        timestamp   = timestamp     ,
    )
    logger_arcs.save()  # Will overwrite the previous arcs log by model_follower.py

    logger_arcs.debug("Arcs information:")
    for id, arc in all_arcs.items():
        logger_arcs.debug (f"  Arc {id}: type {arc.type} from node ({arc.o.i}, {arc.o.t}, {arc.o.l}) to ({arc.d.i}, {arc.d.t}, {arc.d.l}); flow: {x[id]}")
    
    # ----------------------------
    # EV flow in each arcs
    # ----------------------------
    logger_arcs.info("EV flow in arcs:")
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
            logger_arcs.info (f"  Time {curr_time}:")

        logger_arcs.info(f"    Arc {arc_id} ({arc.type.name}): ")
        logger_arcs.info(f"      From (start zone {arc.o.i}, start time {arc.o.t}, start charge {arc.o.l}) to (end zone {arc.d.i}, end time {arc.d.t}, end charge {arc.d.l})")
        logger_arcs.info(f"      Flow: {x.get(arc_id, 0)}")
        if arc.type == ArcType.SERVICE:
            logger_arcs.info(f"      Unserved Demand: {s.get((arc.o.i, arc.d.i, arc.o.t), 0)}")

    
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

    if not math.isclose(total_trips, total_trips_served + total_trips_unserved):
        logger.warning (f"  Total trips requested ({total_trips}) does not match total trips served ({total_trips_served}) + unserved ({total_trips_unserved}).")

    if not math.isclose(total_service_time + total_charge_time + total_relocation_time + total_idle_time, T * num_EVs):
        logger.warning (f"  Total time does not sum up! Total time: {T * num_EVs}, Service time: {total_service_time}, Charge time: {total_charge_time}, Relocation time: {total_relocation_time}, Idle time: {total_idle_time}")

    postprocessing(
        # Follower model parameters
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

        # Network components
        V_set                   = V_set                 ,
        all_arcs                = all_arcs              ,
        type_arcs               = type_arcs             ,
        in_arcs                 = in_arcs               ,
        out_arcs                = out_arcs              ,
        service_arcs_ijt        = service_arcs_ijt      ,
        charge_arcs_it          = charge_arcs_it        ,
        charge_arcs_t           = charge_arcs_t         ,
        valid_travel_demand     = valid_travel_demand   ,
        invalid_travel_demand   = invalid_travel_demand ,
        ZONES                   = ZONES                 ,
        TIMESTEPS               = TIMESTEPS             ,
        LEVELS                  = LEVELS                ,
        AGES                    = AGES                  ,

        # Leader model parameters
        wholesale_elec_price    = wholesale_elec_price  ,
        PENALTY_WEIGHT          = PENALTY_WEIGHT        ,

        # Pricing Variables
        charge_cost_low         = charge_cost_low       ,
        charge_cost_high        = charge_cost_high      ,
        elec_threshold          = elec_threshold        ,  

        # Solutions
        obj                     = obj                   ,
        x                       = x                     ,
        s                       = s                     ,
        u                       = u                     ,
        e                       = e                     ,
        q                       = q                     ,
        service_revenues        = service_revenues      ,
        penalty_costs           = penalty_costs         ,
        charge_costs            = charge_costs          ,

        # Calculated information
        total_service_revenue   = total_service_revenue ,
        total_penalty_cost      = total_penalty_cost    ,
        total_charge_cost       = total_charge_cost     ,

        # Metadata
        timestamp               = timestamp             ,
        file_name               = file_name             ,
        folder_name             = folder_name           ,
        results_name            = results_name          ,
    )


        


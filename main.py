from datetime import datetime
import json
import os
import traceback
import math
import multiprocessing
import time

from logger import Logger
from model_follower import follower_model, follower_graph_builder
from networkClass import Node, Arc, ArcType
from utility import convert_key_types, print_duration
from postprocessing import postprocessing
from bilevel_DE import run_parallel_de
from config_DE import (
    POP_SIZE        ,
    NUM_PROCESSES   ,
    NUM_THREADS     ,
    MAX_ITER        ,
    DIFF_WEIGHT     ,
    CROSS_PROB      ,
    VAR_THRESHOLD   ,
    PENALTY_WEIGHT  ,
    NUM_ANCHORS     ,
    VARS_PER_STEP   ,
    RELAX_STAGE_2
)


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
    model_choice = input ("Enter '1' to run only the follower model (stage 2), or press '2' to run the bilevel optimization model (stage 1 and 2); default is '2': \n").strip()
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

    path_metadata = {
        "file_name"     : file_name     ,
        "folder_name"   : folder_name   ,
        "timestamp"     : timestamp     ,
    }

    # Create necessary directories if they do not exist
    os.makedirs("Logs", exist_ok=True)
    os.makedirs(os.path.join("Logs", folder_name), exist_ok=True)
    os.makedirs("Results", exist_ok=True)
    os.makedirs(os.path.join("Results", folder_name), exist_ok=True)

    # Set up logger
    logger = Logger(
        name        = "main"        , 
        level       = "DEBUG"       , 
        to_console  = True          , 
        **path_metadata             ,
    )
    logger.save()

    results_name = os.path.join("Results", folder_name, f"results_{file_name}_{timestamp}.xlsx")

    logger.info("Starting the EV Fleet Charging Optimization Model...")
    logger.info (f"Test case file: {file_name}.json in folder: {directory if directory != '' else 'Testcases/'}")
    logger.info (f"Results will be saved in: {results_name}")

    logger.info(f"Configurations for Differential Evolution:")
    logger.info(f"  Population Size (POP_SIZE)                              : {POP_SIZE}")
    logger.info(f"  Number of Processes (NUM_PROCESSES)                     : {NUM_PROCESSES}")
    logger.info(f"  Number of Threads per Process (NUM_THREADS)             : {NUM_THREADS}")
    logger.info(f"  Maximum Iterations (MAX_ITER)                           : {MAX_ITER}")
    logger.info(f"  Differential Weight (DIFF_WEIGHT)                       : {DIFF_WEIGHT}")
    logger.info(f"  Crossover Probability (CROSS_PROB)                      : {CROSS_PROB}")
    logger.info(f"  Variance Threshold for Early Stopping (VAR_THRESHOLD)   : {VAR_THRESHOLD}")
    logger.info(f"  Penalty Weight for Leader Fitness (PENALTY_WEIGHT)      : {PENALTY_WEIGHT}")
    logger.info(f"  Number of Anchors (NUM_ANCHORS)                         : {NUM_ANCHORS}")
    logger.info(f"  Variables per Time Step (VARS_PER_STEP)                 : {VARS_PER_STEP}")
    logger.info(f"  Relax Follower Model in Stage 2 (RELAX_STAGE_2)         : {RELAX_STAGE_2}")

    # Load data from the specified JSON file
    processed_data: dict = None
    start_time_load_data = time.time()
    try:
        with open(os.path.join ("Testcases", directory, file_name + ".json"), "r") as F:
            
            logger.info(f"Loading data from {file_name}.json...")
            data = json.load(F)

            # logger.debug(f"Raw data loaded: {data}")
            logger.info ("Raw data loaded. Converting keys in data to appropriate types...")
            processed_data = convert_key_types(data)  # Convert keys to tuples of integers
            # logger.debug(f"Processed data: {processed_data}")

            end_time_load_data = time.time()
            duration_load_data = end_time_load_data - start_time_load_data
            logger.info (f"Data loaded successfully in {print_duration(duration_load_data)} ({duration_load_data:.2f} seconds).")

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

    follower_model_parameters = {
        "N"                     : N                     ,
        "T"                     : T                     ,
        "L"                     : L                     ,
        "W"                     : W                     ,
        "travel_demand"         : travel_demand         ,
        "travel_time"           : travel_time           ,
        "travel_energy"         : travel_energy         ,
        "order_revenue"         : order_revenue         ,
        "penalty"               : penalty               ,
        "L_min"                 : L_min                 ,
        "num_EVs"               : num_EVs               ,
        "num_ports"             : num_ports             ,
        "elec_supplied"         : elec_supplied         ,
        "max_charge_speed"      : max_charge_speed      ,
    }
    leader_model_parameters = {
        "wholesale_elec_price"  : wholesale_elec_price  ,
    }    

    # -----------------------------------------------------------
    # Build the network for the follower model (reusable)
    # -----------------------------------------------------------
    start_time_build_network = time.time()
    network_parameters = follower_graph_builder(
        **follower_model_parameters ,
        **path_metadata             ,
    )
    end_time_build_network = time.time()
    duration_build_network = end_time_build_network - start_time_build_network
    
    # Extract reusable network components
    V_set                  : set[Node]                              = network_parameters["V_set"]
    all_arcs               : dict[int                   , Arc]      = network_parameters["all_arcs"]
    type_arcs              : dict[ArcType               , set[int]] = network_parameters["type_arcs"]
    in_arcs                : dict[Node                  , set[int]] = network_parameters["in_arcs"]
    out_arcs               : dict[Node                  , set[int]] = network_parameters["out_arcs"]
    service_arcs_ijt       : dict[tuple[int, int, int]  , set[int]] = network_parameters["service_arcs_ijt"]
    charge_arcs_it         : dict[tuple[int, int]       , set[int]] = network_parameters["charge_arcs_it"]
    charge_arcs_t          : dict[int                   , set[int]] = network_parameters["charge_arcs_t"]
    valid_travel_demand    : dict[tuple[int, int, int]  , int]      = network_parameters["valid_travel_demand"]
    invalid_travel_demand  : set[tuple[int, int, int]]              = network_parameters["invalid_travel_demand"]
    ZONES                  : list[int]                              = network_parameters["ZONES"]
    TIMESTEPS              : list[int]                              = network_parameters["TIMESTEPS"]
    LEVELS                 : list[int]                              = network_parameters["LEVELS"]
    AGES                   : list[int]                              = network_parameters["AGES"]

    logger.info(f"Network built in {print_duration(duration_build_network)} ({duration_build_network:.2f} seconds).")
    
    # -----------------------------------------------------------
    # Stage 1: Run the bilevel optimization on relaxed model
    # -----------------------------------------------------------
    if model_choice == "1":
        logger.info("Stage 1 skipped as per user choice. Running only the follower model (Stage 2)...")
        try:
            charge_cost_low     : dict[int, float]                  = processed_data["charge_cost_low"]       # a_t
            charge_cost_high    : dict[int, float]                  = processed_data["charge_cost_high"]      # b_t
            elec_threshold      : dict[int, int]                    = processed_data["elec_threshold"]        # r_t

            charge_price_parameters = {
                "charge_cost_low"    : charge_cost_low     ,
                "charge_cost_high"   : charge_cost_high    ,
                "elec_threshold"     : elec_threshold      ,
            }
        except KeyError as e:
            logger.error(f"Missing required pricing/threshold parameter in input data for follower model: {e}")
            exit(1)
    else:
        logger.info("Stage 1: Starting bilevel optimization using Differential Evolution...")
        start_time_stage1 = time.time()

        try:
            charge_price_parameters = run_parallel_de(
                **follower_model_parameters         ,
                **leader_model_parameters           ,
                **network_parameters                ,
                **path_metadata                     ,
            )   
        except Exception as e:
            logger.error(f"Failure in Stage 1: {e}")
            logger.error(traceback.format_exc())
            exit(1)

        end_time_stage1 = time.time()
        duration_stage1 = end_time_stage1 - start_time_stage1
        logger.info(f"Stage 1 completed in {print_duration(duration_stage1)} ({duration_stage1:.2f} seconds).")

        charge_cost_low     : dict[int, float]                  = charge_price_parameters["charge_cost_low"]       # a_t
        charge_cost_high    : dict[int, float]                  = charge_price_parameters["charge_cost_high"]      # b_t
        elec_threshold      : dict[int, int]                    = charge_price_parameters["elec_threshold"]        # r_t


    # ----------------------------------------------------------------------------------
    # Stage 2: Run the original follower model with the obtained pricing and threshold
    # ----------------------------------------------------------------------------------
    logger.info("Stage 2: Running follower model with obtained pricing and threshold...")
    start_time_stage2 = time.time()

    try:
        solutions_stage2 = follower_model(
            **follower_model_parameters                     ,
            **network_parameters                            ,
            **charge_price_parameters                       ,
            **path_metadata                                 ,
            # Metadata
            NUM_THREADS = multiprocessing.cpu_count() - 1   ,   # use all available cores minus one for the final run
        )
    except Exception as e:
        logger.error(f"Failure in Stage 2: {e}")
        logger.error(traceback.format_exc())
        exit(1)
        
    end_time_stage2 = time.time()
    duration_stage2 = end_time_stage2 - start_time_stage2
    logger.info(f"Stage 2 completed in {print_duration(duration_stage2)} ({duration_stage2:.2f} seconds).")


    # Extract variables and sets from the output
    obj             : float                                     = solutions_stage2["obj"]
    x               : dict[int, float]                          = solutions_stage2["x"]
    s               : dict[tuple[int, int, int]     , float]    = solutions_stage2["s"]
    u               : dict[tuple[int, int, int, int], float]    = solutions_stage2["u"]
    e               : dict[tuple[int, int, int]     , float]    = solutions_stage2["e"]
    q               : dict[int                      , float]    = solutions_stage2["q"]


    # ----------------------------
    # Arcs Information
    # ----------------------------    
    logger_arcs = Logger(
        name        = "arcs"        , 
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
    service_revenues        : dict[int, float] = {
        t: sum (x[e_id] * all_arcs[e_id].revenue
            for i in ZONES
            for j in ZONES
            for e_id in service_arcs_ijt.get((i, j, t), set())
        )
        for t in TIMESTEPS
    }
    penalty_costs           : dict[int, float] = {
        t: sum (e[(i, j, t)] * penalty.get((i, j, t), 0)
            for i in ZONES
            for j in ZONES
        ) + (sum (s[(i, j, t)] * penalty.get((i, j, t), 0)
            for i in ZONES
            for j in ZONES
            ) if t == T else 0
        )
        for t in TIMESTEPS
    }
    charge_costs            : dict[int, float] = {
        t: sum (x[e_id] * all_arcs[e_id].charge_speed
            for e_id in charge_arcs_t.get(t, set())
        ) * charge_cost_low.get(t, 0) \
        + q[t] * charge_cost_high.get(t, 0)
        for t in TIMESTEPS
    }
    total_service_revenue   : float = sum(service_revenues.values())
    total_penalty_cost      : float = sum(penalty_costs.values())
    total_charge_cost       : float = sum(charge_costs.values())

    logger.info("Model results:")
    logger.info(f"  Objective value: {obj:.2f}")
    logger.info(f"  Total service revenue: {total_service_revenue:.2f}")
    logger.info(f"  Total penalty cost: {total_penalty_cost:.2f}")
    logger.info(f"  Total charge cost: {total_charge_cost:.2f}")    

    if not math.isclose(obj, total_service_revenue - total_penalty_cost - total_charge_cost):
        logger.warning (f"  Objective value ({obj:.2f}) does not match total service revenue - total penalty cost - total charge cost ({total_service_revenue - total_penalty_cost - total_charge_cost:.2f}).")
    

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


    # ---------------------------------
    # Postprocessing and save results
    # ---------------------------------
    postprocessing(
        **follower_model_parameters                     ,
        **leader_model_parameters                       ,
        **network_parameters                            ,
        **charge_price_parameters                       ,
        **solutions_stage2                              ,
        **path_metadata                                 ,

        # Calculated information
        service_revenues        = service_revenues      ,
        penalty_costs           = penalty_costs         ,
        charge_costs            = charge_costs          ,
        total_service_revenue   = total_service_revenue ,
        total_penalty_cost      = total_penalty_cost    ,
        total_charge_cost       = total_charge_cost     ,

        # Metadata
        results_name            = results_name          ,
    )


        


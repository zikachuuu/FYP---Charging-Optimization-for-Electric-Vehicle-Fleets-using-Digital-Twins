import numpy as np
import numpy.typing as npt
import time
import multiprocessing
import os
from functools import partial
import traceback

from model_leader import leader_model
from model_follower import follower_model
from networkClass import Node, Arc, ArcType
from logger import Logger, LogListener
from exceptions import OptimizationError

def _precompute_interpolation_matrix(
        T               : int                   , 
        NUM_ANCHORS     : int                   , 
        anchor_indices  : npt.NDArray[np.int_]  ,
    ) -> npt.NDArray[np.float64]:
    """
    PRE-COMPUTE THIS ONCE OUTSIDE THE LOOP
    Creates a (T+1, num_anchors) matrix M such that: Full_Trajectory = M @ Anchors
    """
    # Create interpolation matrix
    # The rows correspond to time steps (0 to T)
    # The columns correspond to anchor points
    # The values are the weights for linear interpolation (each row sums to 1)
    # eg. for T=5, num_anchors=3 at indices [0, 2, 5]:
    #       | 1.0   0.0     0.0  |  # t=0
    #       | 0.5   0.5     0.0  |  # t=1
    #   M = | 0.0   1.0     0.0  |  # t=2
    #       | 0.0  0.6667 0.3333 |  # t=3
    #       | 0.0  0.3333 0.6667 |  # t=4
    #       | 0.0   0.0     1.0  |  # t=5
    M: npt.NDArray[np.float64] = np.zeros((T+1, NUM_ANCHORS))
    
    for i in range(NUM_ANCHORS - 1):

        anchor_start    = anchor_indices[i]         # eg 0
        anchor_end      = anchor_indices[i+1]       # eg 2
        segment_len     = anchor_end - anchor_start # eg 2 - 0 = 2

        if segment_len == 0:
            continue  # avoid division by zero for duplicate anchors
            
        slope = 1.0 / segment_len
        
        # Fill the matrix rows corresponding to this time segment
        for t in range(anchor_start, anchor_end + 1): # +1 to include end for continuity
            local_x = t - anchor_start
            weight_next = local_x * slope
            weight_prev = 1.0 - weight_next
            
            # M[t, i] is weight of anchor i
            # M[t, i+1] is weight of anchor i+1
            M[t, i] = weight_prev
            M[t, i+1] = weight_next
            
    return M


def _expand_trajectory (
        candidate_flat  : npt.NDArray[np.float64]   , 
        M               : npt.NDArray[np.float64]   , 
        VARS_PER_STEP   : int                       ,
        lower_bounds_a  : npt.NDArray[np.float64]   ,
        lower_bounds_b  : npt.NDArray[np.float64]   ,
        lower_bounds_r  : npt.NDArray[np.float64]   ,
        upper_bounds_r  : npt.NDArray[np.float64]   ,
        TIMESTEPS       : list[int]                 ,
    ) -> dict[str, dict[int, float]]:
    # compressed shape: (num_anchors * vars_per_step, )
    # reshape to (num_anchors, vars_per_step)
    anchors = candidate_flat.reshape(-1, VARS_PER_STEP)
    
    # (T+1, num_anchors) @ (num_anchors, 3) -> (T+1, 3)
    full_trajectory = M @ anchors

    # Enforce lower and upper bounds
    full_trajectory[:, 0] = np.maximum(full_trajectory[:, 0], lower_bounds_a)  # a_t
    full_trajectory[:, 1] = np.maximum(full_trajectory[:, 1], lower_bounds_b)  # b_t
    full_trajectory[:, 2] = np.clip(full_trajectory[:, 2], lower_bounds_r, upper_bounds_r)  # r_t

    # charge_cost_low (a_t): first column
    # charge_cost_high (b_t): second column
    # elec_threshold (r_t): third column
    charge_cost_low     : dict[int, float]  = {t: full_trajectory[t, 0]         for t in TIMESTEPS}  # a_t
    charge_cost_high    : dict[int, float]  = {t: full_trajectory[t, 1]         for t in TIMESTEPS}  # b_t
    elec_threshold      : dict[int, int]    = {t: round(full_trajectory[t, 2])  for t in TIMESTEPS}  # r_t, ensure integer

    return {
        "charge_cost_low"    : charge_cost_low        ,
        "charge_cost_high"   : charge_cost_high       ,
        "elec_threshold"     : elec_threshold         ,
    }


def _reference_candidate(
        **kwargs
    ):
    """
    Generate the reference variance value by solving the follower problem with minimum prices.
    """
    # ----------------------------
    # Parameters
    # ----------------------------
    # Follower model parameters
    N                       : int                                   = kwargs["N"]                       # number of operation zones (1, ..., N)
    T                       : int                                   = kwargs["T"]                       # termination time of daily operations (0, ..., T)
    L                       : int                                   = kwargs["L"]                       # max SoC level (all EVs start at this level) (0, ..., L)
    W                       : int                                   = kwargs["W"]                       # maximum time intervals a passenger will wait for a ride (0, ..., W-1; demand expires at W)
    travel_demand           : dict[tuple[int, int, int] , int]      = kwargs["travel_demand"]           # travel demand from zone i to j at starting at time t
    travel_time             : dict[tuple[int, int, int] , int]      = kwargs["travel_time"]             # travel time from i to j at starting at time t
    travel_energy           : dict[tuple[int, int]      , int]      = kwargs["travel_energy"]           # energy consumed for trip from zone i to j
    order_revenue           : dict[tuple[int, int, int] , float]    = kwargs["order_revenue"]           # order revenue for each trip served from i to j at time t
    penalty                 : dict[tuple[int, int, int] , float]    = kwargs["penalty"]                 # penalty cost for each unserved trip from i to j at time t
    L_min                   : int                                   = kwargs["L_min"]                   # min SoC level all EV must end with at the end of the daily operations
    num_EVs                 : int                                   = kwargs["num_EVs"]                 # total number of EVs in the fleet 
    num_ports               : dict[int                  , int]      = kwargs["num_ports"]               # number of charging ports in each zone
    elec_supplied           : dict[tuple[int, int]      , int]      = kwargs["elec_supplied"]           # electricity supplied (in SoC levels) at zone i at time t
    max_charge_speed        : int                                   = kwargs["max_charge_speed"]        # max charging speed (in SoC levels) of one EV in one time step

    # Network components
    V_set                   : set[Node]                             = kwargs["V_set"]
    all_arcs                : dict[int                  , Arc]      = kwargs["all_arcs"]
    type_arcs               : dict[ArcType              , set[int]] = kwargs["type_arcs"]
    in_arcs                 : dict[Node                 , set[int]] = kwargs["in_arcs"]
    out_arcs                : dict[Node                 , set[int]] = kwargs["out_arcs"]
    service_arcs_ijt        : dict[tuple[int, int, int] , set[int]] = kwargs["service_arcs_ijt"]
    charge_arcs_it          : dict[tuple[int, int]      , set[int]] = kwargs["charge_arcs_it"]
    charge_arcs_t           : dict[int                  , set[int]] = kwargs["charge_arcs_t"]
    valid_travel_demand     : dict[tuple[int, int, int] , int]      = kwargs["valid_travel_demand"]
    invalid_travel_demand   : set[tuple[int, int, int]]             = kwargs["invalid_travel_demand"]
    ZONES                   : list[int]                             = kwargs["ZONES"]
    TIMESTEPS               : list[int]                             = kwargs["TIMESTEPS"]
    LEVELS                  : list[int]                             = kwargs["LEVELS"]
    AGES                    : list[int]                             = kwargs["AGES"]

    # Pricing Variables
    charge_cost_low         : dict[int                  , float]    = kwargs["charge_cost_low"]         # a_t
    charge_cost_high        : dict[int                  , float]    = kwargs["charge_cost_high"]        # b_t
    elec_threshold          : dict[int                  , int]      = kwargs["elec_threshold"]          # r_t

    # Metadata
    timestamp               : str                                   = kwargs["timestamp"]               # timestamp for logging
    file_name               : str                                   = kwargs["file_name"]               # filename for logging
    folder_name             : str                                   = kwargs["folder_name"]             # folder name for logging
    logger                  : Logger                                = kwargs["logger"]                  # logger instance

    logger.info("Solving follower problem for reference candidate with minimum prices...")

    # ----------------------------
    # Solve Follower Problem
    # ----------------------------
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

            # Pricing Variables
            charge_cost_low         = charge_cost_low       ,
            charge_cost_high        = charge_cost_high      ,
            elec_threshold          = elec_threshold        ,  

            # Metadata
            relaxed                 = True                  ,
            timestamp               = timestamp             ,
            file_name               = file_name             ,
            folder_name             = folder_name           ,
            logger                  = logger                ,
        )
    except OptimizationError as e:
        raise OptimizationError("Failed to solve follower problem for reference candidate.", details=e) from e
    except Exception as e:
        raise Exception("Unexpected error when solving follower problem for reference candidate.") from e

    logger.info("Reference candidate solved successfully.")

    # Extract variables and sets from the follower_outputs    
    obj             : float                                     = follower_outputs["obj"]
    x               : dict[int, float]                          = follower_outputs["x"]
    s               : dict[tuple[int, int, int]     , float]    = follower_outputs["s"]
    u               : dict[tuple[int, int, int, int], float]    = follower_outputs["u"]
    e               : dict[tuple[int, int, int]     , float]    = follower_outputs["e"]
    q               : dict[int                      , float]    = follower_outputs["q"]
    service_revenues: dict[int                      , float]    = follower_outputs["service_revenues"]
    penalty_costs   : dict[int                      , float]    = follower_outputs["penalty_costs"]
    charge_costs    : dict[int                      , float]    = follower_outputs["charge_costs"]

    # Calculate electricity consumption at each time step using vectorized operations
    electricity_usage: npt.NDArray[np.float64] = np.zeros(T + 1)

    for t in TIMESTEPS:
        # Calculate total electricity used at time t
        for e_id in charge_arcs_t.get(t, set()):
            arc = all_arcs[e_id]
            
            # Electricity used = number of EVs * charge amount
            charge_amount = arc.d.l - arc.o.l  # SoC levels charged
            electricity_usage[t] += x[e_id] * charge_amount

    # Calculate variance of electricity consumption using numpy
    usage_vector    : npt.NDArray[np.float64]   = electricity_usage[1:T]  # exclude time 0 and T
    variance        : float                     = np.var(usage_vector, ddof=0) if len(usage_vector) > 1 else 0.0

    return variance


def _evaluate_single_candidate(
        candidate_flat,
        **kwargs
    ):
    """
    Calculate and returns the penalty and variance for a single candidate solution.
    """    
    # ----------------------------
    # Parameters
    # ----------------------------
    # Follower model parameters
    N                       : int                                   = kwargs["N"]                       # number of operation zones (1, ..., N)
    T                       : int                                   = kwargs["T"]                       # termination time of daily operations (0, ..., T)
    L                       : int                                   = kwargs["L"]                       # max SoC level (all EVs start at this level) (0, ..., L)
    W                       : int                                   = kwargs["W"]                       # maximum time intervals a passenger will wait for a ride (0, ..., W-1; demand expires at W)
    travel_demand           : dict[tuple[int, int, int] , int]      = kwargs["travel_demand"]           # travel demand from zone i to j at starting at time t
    travel_time             : dict[tuple[int, int, int] , int]      = kwargs["travel_time"]             # travel time from i to j at starting at time t
    travel_energy           : dict[tuple[int, int]      , int]      = kwargs["travel_energy"]           # energy consumed for trip from zone i to j
    order_revenue           : dict[tuple[int, int, int] , float]    = kwargs["order_revenue"]           # order revenue for each trip served from i to j at time t
    penalty                 : dict[tuple[int, int, int] , float]    = kwargs["penalty"]                 # penalty cost for each unserved trip from i to j at time t
    L_min                   : int                                   = kwargs["L_min"]                   # min SoC level all EV must end with at the end of the daily operations
    num_EVs                 : int                                   = kwargs["num_EVs"]                 # total number of EVs in the fleet 
    num_ports               : dict[int                  , int]      = kwargs["num_ports"]               # number of charging ports in each zone
    elec_supplied           : dict[tuple[int, int]      , int]      = kwargs["elec_supplied"]           # electricity supplied (in SoC levels) at zone i at time t
    max_charge_speed        : int                                   = kwargs["max_charge_speed"]        # max charging speed (in SoC levels) of one EV in one time step

    # Network components
    V_set                   : set[Node]                             = kwargs["V_set"]
    all_arcs                : dict[int                  , Arc]      = kwargs["all_arcs"]
    type_arcs               : dict[ArcType              , set[int]] = kwargs["type_arcs"]
    in_arcs                 : dict[Node                 , set[int]] = kwargs["in_arcs"]
    out_arcs                : dict[Node                 , set[int]] = kwargs["out_arcs"]
    service_arcs_ijt        : dict[tuple[int, int, int] , set[int]] = kwargs["service_arcs_ijt"]
    charge_arcs_it          : dict[tuple[int, int]      , set[int]] = kwargs["charge_arcs_it"]
    charge_arcs_t           : dict[int                  , set[int]] = kwargs["charge_arcs_t"]
    valid_travel_demand     : dict[tuple[int, int, int] , int]      = kwargs["valid_travel_demand"]
    invalid_travel_demand   : set[tuple[int, int, int]]             = kwargs["invalid_travel_demand"]
    ZONES                   : list[int]                             = kwargs["ZONES"]
    TIMESTEPS               : list[int]                             = kwargs["TIMESTEPS"]
    LEVELS                  : list[int]                             = kwargs["LEVELS"]
    AGES                    : list[int]                             = kwargs["AGES"]

    # Leader model parameters
    wholesale_elec_price    : dict[int                  , float]    = kwargs["wholesale_elec_price"]    # wholesale electricity price at time t
    PENALTY_WEIGHT          : float                                 = kwargs["PENALTY_WEIGHT"]          # penalty weight for high a_t and b_t
    reference_variance      : float                                 = kwargs["reference_variance"]      # reference variance for normalization

    # Pricing Variable bounds
    lower_bounds_a          : npt.NDArray[np.float64]               = kwargs["lower_bounds_a"]          # lower bounds for a_t
    lower_bounds_b          : npt.NDArray[np.float64]               = kwargs["lower_bounds_b"]          # lower bounds for b_t
    lower_bounds_r          : npt.NDArray[np.float64]               = kwargs["lower_bounds_r"]          # lower bounds for r_t
    upper_bounds_r          : npt.NDArray[np.float64]               = kwargs["upper_bounds_r"]          # upper bounds for r_t
    
    # DE parameters
    VARS_PER_STEP           : int                                   = kwargs["VARS_PER_STEP"]           # number of variables per time step (3: a_t, b_t, r_t)
    M                       : npt.NDArray[np.float64]               = kwargs["M"]                       # interpolation matrix

    # Metadata
    timestamp               : str                                   = kwargs["timestamp"]               # timestamp for logging
    file_name               : str                                   = kwargs["file_name"]               # filename for logging
    folder_name             : str                                   = kwargs["folder_name"]             # folder name for logging
    log_queue               : multiprocessing.Queue                 = kwargs["log_queue"]               # multiprocessing log queue

    logger = Logger (
        f"Worker_{os.getpid()}"     ,    
        level       = "DEBUG"       ,
        to_console  = False         ,
        folder_name = folder_name   ,
        file_name   = file_name     ,
        timestamp   = timestamp     ,
        queue       = log_queue     ,
    )

    solutions = _expand_trajectory(
        candidate_flat  = candidate_flat    ,
        M               = M                 ,
        VARS_PER_STEP   = VARS_PER_STEP     ,
        lower_bounds_a  = lower_bounds_a    ,
        lower_bounds_b  = lower_bounds_b    ,
        lower_bounds_r  = lower_bounds_r    ,
        upper_bounds_r  = upper_bounds_r    ,   
        TIMESTEPS       = TIMESTEPS         ,
    )
    charge_cost_low     = solutions["charge_cost_low"]
    charge_cost_high    = solutions["charge_cost_high"]
    elec_threshold      = solutions["elec_threshold"]

    logger.info("Solving follower problem for candidate...")

    # ----------------------------
    # Solve Follower Problem
    # ----------------------------
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

            # Pricing Variables
            charge_cost_low         = charge_cost_low       ,
            charge_cost_high        = charge_cost_high      ,
            elec_threshold          = elec_threshold        ,  

            # Metadata
            relaxed                 = True                  ,
            timestamp               = timestamp             ,
            file_name               = file_name             ,
            folder_name             = folder_name           ,
            logger                  = logger                ,
        )
    except OptimizationError as e:
        raise OptimizationError("Failed to solve follower problem for candidate.", details=e) from e
    except Exception as e:
        raise Exception("Unexpected error when solving follower problem for candidate.") from e

    logger.info("Follower problem for candidate solved successfully.")

    # Extract variables and sets from the follower_outputs    
    obj             : float                                     = follower_outputs["obj"]
    x               : dict[int, float]                          = follower_outputs["x"]
    s               : dict[tuple[int, int, int]     , float]    = follower_outputs["s"]
    u               : dict[tuple[int, int, int, int], float]    = follower_outputs["u"]
    e               : dict[tuple[int, int, int]     , float]    = follower_outputs["e"]
    q               : dict[int                      , float]    = follower_outputs["q"]
    service_revenues: dict[int                      , float]    = follower_outputs["service_revenues"]
    penalty_costs   : dict[int                      , float]    = follower_outputs["penalty_costs"]
    charge_costs    : dict[int                      , float]    = follower_outputs["charge_costs"]

    logger.info("Solving leader problem for candidate...")

    # ----------------------------
    # Solve Leader Problem
    # ----------------------------
    leader_outputs = leader_model(
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
        reference_variance      = reference_variance    ,

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

        # Metadata
        timestamp               = timestamp             ,
        file_name               = file_name             ,
        folder_name             = folder_name           ,
        logger                  = logger                ,
    )

    logger.info("Leader problem for candidate solved successfully.")

    return (
        leader_outputs["fitness"],
        leader_outputs["variance"],
        leader_outputs["variance_ratio"],
        leader_outputs["percentage_price_increase"]
    )


def run_parallel_de(
        **kwargs
    ):
    """
    Runs Differential Evolution in parallel using multiprocessing.
    """
    # ----------------------------
    # Parameters
    # ----------------------------
    # Follower model parameters
    N                       : int                                   = kwargs["N"]                       # number of operation zones (1, ..., N)
    T                       : int                                   = kwargs["T"]                       # termination time of daily operations (0, ..., T)
    L                       : int                                   = kwargs["L"]                       # max SoC level (all EVs start at this level) (0, ..., L)
    W                       : int                                   = kwargs["W"]                       # maximum time intervals a passenger will wait for a ride (0, ..., W-1; demand expires at W)
    travel_demand           : dict[tuple[int, int, int] , int]      = kwargs["travel_demand"]           # travel demand from zone i to j at starting at time t
    travel_time             : dict[tuple[int, int, int] , int]      = kwargs["travel_time"]             # travel time from i to j at starting at time t
    travel_energy           : dict[tuple[int, int]      , int]      = kwargs["travel_energy"]           # energy consumed for trip from zone i to j
    order_revenue           : dict[tuple[int, int, int] , float]    = kwargs["order_revenue"]           # order revenue for each trip served from i to j at time t
    penalty                 : dict[tuple[int, int, int] , float]    = kwargs["penalty"]                 # penalty cost for each unserved trip from i to j at time t
    L_min                   : int                                   = kwargs["L_min"]                   # min SoC level all EV must end with at the end of the daily operations
    num_EVs                 : int                                   = kwargs["num_EVs"]                 # total number of EVs in the fleet 
    num_ports               : dict[int                  , int]      = kwargs["num_ports"]               # number of charging ports in each zone
    elec_supplied           : dict[tuple[int, int]      , int]      = kwargs["elec_supplied"]           # electricity supplied (in SoC levels) at zone i at time t
    max_charge_speed        : int                                   = kwargs["max_charge_speed"]        # max charging speed (in SoC levels) of one EV in one time step
    
    # Network components
    V_set                   : set[Node]                             = kwargs["V_set"]
    all_arcs                : dict[int                  , Arc]      = kwargs["all_arcs"]
    type_arcs               : dict[ArcType              , set[int]] = kwargs["type_arcs"]
    in_arcs                 : dict[Node                 , set[int]] = kwargs["in_arcs"]
    out_arcs                : dict[Node                 , set[int]] = kwargs["out_arcs"]
    service_arcs_ijt        : dict[tuple[int, int, int] , set[int]] = kwargs["service_arcs_ijt"]
    charge_arcs_it          : dict[tuple[int, int]      , set[int]] = kwargs["charge_arcs_it"]
    charge_arcs_t           : dict[int                  , set[int]] = kwargs["charge_arcs_t"]
    valid_travel_demand     : dict[tuple[int, int, int] , int]      = kwargs["valid_travel_demand"]
    invalid_travel_demand   : set[tuple[int, int, int]]             = kwargs["invalid_travel_demand"]
    ZONES                   : list[int]                             = kwargs["ZONES"]
    TIMESTEPS               : list[int]                             = kwargs["TIMESTEPS"]
    LEVELS                  : list[int]                             = kwargs["LEVELS"]
    AGES                    : list[int]                             = kwargs["AGES"]

    # Leader model parameters
    wholesale_elec_price    : dict[int                  , float]    = kwargs["wholesale_elec_price"]    # wholesale electricity price at time t
    PENALTY_WEIGHT          : float                                 = kwargs["PENALTY_WEIGHT"]          # penalty weight for high a_t and b_t
    
    # DE parameters
    POP_SIZE                : int                                   = kwargs["POP_SIZE"]                # population size for DE
    MAX_ITER                : int                                   = kwargs["MAX_ITER"]                # max iterations for DE
    DIFF_WEIGHT             : float                                 = kwargs["DIFF_WEIGHT"]             # differential weight
    CROSS_PROB              : float                                 = kwargs["CROSS_PROB"]              # crossover probability
    VAR_THRESHOLD           : float                                 = kwargs["VAR_THRESHOLD"]           # variance threshold for stopping criteria
    NUM_ANCHORS             : int                                   = kwargs["NUM_ANCHORS"]             # number of anchors for DE
    VARS_PER_STEP           : int                                   = kwargs["VARS_PER_STEP"]           # number of variables per time step (3: a_t, b_t, r_t)

    # Metadata
    timestamp               : str                                   = kwargs["timestamp"]               # timestamp for logging
    file_name               : str                                   = kwargs["file_name"]               # filename for logging
    folder_name             : str                                   = kwargs["folder_name"]             # folder name for logging


    # ----------------------------
    # Logger Setup
    # ----------------------------
    # 1. Create the Queue using Manager (Safest for Pools)
    manager = multiprocessing.Manager()
    log_queue = manager.Queue()

    # 2. Start the Listener Process
    listener = LogListener(
        "stage1_MP"         ,
        folder_name         , 
        file_name           , 
        timestamp           ,
        log_queue           ,
        to_console = True   ,
    )
    listener.start()

    try:
        # Logger for this process (no need multiprocessing logger here)
        logger = Logger (
            "bilevel_DE"                ,
            level       = "DEBUG"       , 
            to_console  = True          , 
            folder_name = folder_name   , 
            file_name   = file_name     , 
            timestamp   = timestamp     ,       
        )
        logger.save()
        logger.info("Parameters loaded successfully")


        # ----------------------------
        # Population Initialization
        # ----------------------------
        # Using numpy array is much faster for numerical operations
        wholesale_elec_price_arr: npt.NDArray[np.float64] = np.array([wholesale_elec_price.get(t, 0) for t in TIMESTEPS])

        # Lower bounds
        lower_bounds_a          : npt.NDArray[np.float64] = wholesale_elec_price_arr.copy()          # a_t >= wholesale price
        lower_bounds_b          : npt.NDArray[np.float64] = np.zeros(T + 1)                          # b_t >= 0
        lower_bounds_r          : npt.NDArray[np.float64] = np.zeros(T + 1)                          # r_t >= 0

        # Upper bounds
        upper_bounds_a_init     : npt.NDArray[np.float64] = wholesale_elec_price_arr * 2.0                                                  # a_t <= 2 * wholesale price (initial, can be exceeded during evolution)
        upper_bounds_b_init     : npt.NDArray[np.float64] = wholesale_elec_price_arr * 1.0                                                  # b_t <= wholesale price (initial, can be exceeded during evolution)
        upper_bounds_r          : npt.NDArray[np.float64] = np.array([sum(elec_supplied.get((i,t), 0) for i in ZONES) for t in TIMESTEPS])  # r_t <= total electricity supplied at time t (hard limit)

        # a_t and b_t can be unbounded in theory, but we set a very high upper bound for numerical stability
        upper_bounds_a_final    : npt.NDArray[np.float64] = wholesale_elec_price_arr * 10.0
        upper_bounds_b_final    : npt.NDArray[np.float64] = wholesale_elec_price_arr * 10.0

        logger.info("Bounds for decision variables set successfully")

        # anchor indices (dimensionality reduction)
        # instead of optimizing for all T time steps, we pick num_anchors key points and interpolate the rest
        anchor_indices          : npt.NDArray[np.int_]   = np.linspace(0, T, NUM_ANCHORS, dtype=int)
        optimization_dims       : int                    = NUM_ANCHORS * VARS_PER_STEP
        M                       : npt.NDArray[np.float64]= _precompute_interpolation_matrix(T, NUM_ANCHORS, anchor_indices)

        logger.info(f"Interpolation matrix precomputed with shape {M.shape}")

        # array of bounds for anchored variables only
        # Flattened in the order of [a_0, b_0, r_0, a_1, b_1, r_1, ..., a_T, b_T, r_T]
        lower_bounds            : npt.NDArray[np.float64] = np.column_stack(
            (lower_bounds_a, lower_bounds_b, lower_bounds_r)
        )[anchor_indices].flatten()

        upper_bounds_init       : npt.NDArray[np.float64] = np.column_stack(
            (upper_bounds_a_init, upper_bounds_b_init, upper_bounds_r)
        )[anchor_indices].flatten()

        upper_bounds_final      : npt.NDArray[np.float64] = np.column_stack(
            (upper_bounds_a_final, upper_bounds_b_final, upper_bounds_r)
        )[anchor_indices].flatten()

        # Initialize Population
        # each row is a candidate represented as a 1D array of length optimization_dims
        # e.g. [a_0, b_0, r_0, a_1, b_1, r_1, ..., a_T, b_T, r_T]
        # number of rows = pop_size
        population              : npt.NDArray[np.float64] = np.random.uniform (
            low     = lower_bounds, 
            high    = upper_bounds_init, 
            size    = (POP_SIZE, optimization_dims)
        )
        logger.info(f"DE population initialized with size {population.shape}")

        # Ensure that population contain one candidate with all variables at lower bounds
        # This ensures that setting minimum prices is always considered
        population[0,:] = lower_bounds.copy()


        # ----------------------------
        # DE Setup
        # ----------------------------
        # Obtain reference variance from the candidate with lowest possible charge costs
        # We do not use lower_bounds.copy() as our candidate here as it contain only the anchor points
        # after linear interpolation, some time steps may end up above the wholesale price
        lowest_charge_cost_low  = wholesale_elec_price.copy()  # shallow copy is ok as the values are floats
        lowest_charge_cost_high = {t: 0.0 for t in TIMESTEPS}
        lowest_elec_threshold   = {t: 0 for t in TIMESTEPS}

        try:
            reference_variance = _reference_candidate(
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

                # Pricing Variables
                charge_cost_low         = lowest_charge_cost_low ,
                charge_cost_high        = lowest_charge_cost_high,
                elec_threshold          = lowest_elec_threshold  ,

                # Metadata
                timestamp               = timestamp             ,
                file_name               = file_name             ,
                folder_name             = folder_name           ,
                logger                  = logger                ,
            )
        except OptimizationError as e:
            logger.error("Failed to obtain reference variance from reference candidate.")
            raise e
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise e

        logger.info(f"Reference variance obtained: {reference_variance:.5f}")

        if reference_variance < VAR_THRESHOLD:
            logger.info("Reference variance is already below the variance threshold. No optimization needed.")
            return {
                "charge_cost_low"    : lowest_charge_cost_low ,
                "charge_cost_high"   : lowest_charge_cost_high,
                "elec_threshold"     : lowest_elec_threshold  ,
            }

        # Create a partial function with fixed parameters for the worker function
        # Now the worker function only needs the candidate vector as input
        evaluate_single_candidate_worker = partial(
            _evaluate_single_candidate,

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
            reference_variance      = reference_variance    ,
            
            # Pricing Variable bounds
            lower_bounds_a          = lower_bounds_a        ,
            lower_bounds_b          = lower_bounds_b        ,
            lower_bounds_r          = lower_bounds_r        ,
            upper_bounds_r          = upper_bounds_r        ,

            # DE parameters
            VARS_PER_STEP           = VARS_PER_STEP         ,
            M                       = M                     ,

            # Metadata
            timestamp               = timestamp             ,
            file_name               = file_name             ,
            folder_name             = folder_name           ,
            log_queue               = log_queue             ,
        )


        # ----------------------------
        # DE Main Loop
        # ----------------------------
        start_time = time.time()

        # Initial Evaluation
        # We use a Pool to run all candidates in parallel
        # Use processes=cpu_count()-1 to leave one core free
        num_cores = max(1, multiprocessing.cpu_count() - 1)
        
        logger.info(f"Initializing pool with {num_cores} cores...")

        with multiprocessing.Pool(processes=num_cores) as pool:

            # Calculate initial fitness (Parallel)
            # Runs the evaluate_single_candidate function on each candidate in the population in parallel
            try:
                results     : npt.NDArray[np.float64]    = np.array(pool.map(evaluate_single_candidate_worker, population))
            except Exception as e:
                logger.error("An error occurred during the initial evaluation of candidates.")
                raise e

            # Extract fitness and variance (first and second columns of each row)
            fitnesses                   : npt.NDArray[np.float64]    = results[:,0]
            variances                   : npt.NDArray[np.float64]    = results[:,1]
            variance_ratios             : npt.NDArray[np.float64]    = results[:,2]
            percentage_price_increases  : npt.NDArray[np.float64]    = results[:,3]

            init_end_time       : float = time.time()

            # Check if any candidate meets variance threshold
            # if so, early stop by taking the candidate with the lowest fitness among them
            meet_threshold: npt.NDArray[np.bool_] = variances <= VAR_THRESHOLD

            if np.any(meet_threshold):
                qualified_indices           : npt.NDArray[np.int_]      = np.where(meet_threshold)[0]
                qualified_fitnesses         : npt.NDArray[np.float64]   = fitnesses[qualified_indices]
                best_idx_within_qualified   : int                       = qualified_indices[np.argmin(qualified_fitnesses)]
                best_vector                 : npt.NDArray[np.float64]   = population[best_idx_within_qualified].copy()

                logger.info(f"Early stopping at initialization with Var = {variances[best_idx_within_qualified]:.5f}, \
                                                                    Fitness = {fitnesses[best_idx_within_qualified]:.5f}, \
                                                                    Var Ratio =  {variance_ratios[best_idx_within_qualified]:.5f}, \
                                                                    Percentage Price Increase = {percentage_price_increases[best_idx_within_qualified]:.5f}%"
                )
                logger.info(f"  Time taken: {init_end_time - start_time:.1f}s")

                return _expand_trajectory(
                    best_vector,
                    M               = M                   ,
                    VARS_PER_STEP   = VARS_PER_STEP       ,
                    lower_bounds_a  = lower_bounds_a      ,
                    lower_bounds_b  = lower_bounds_b      ,
                    lower_bounds_r  = lower_bounds_r      ,
                    upper_bounds_r  = upper_bounds_r      ,
                    TIMESTEPS       = TIMESTEPS           ,
                )

            # Track both the candidate with best variance (and its fitness), and the candidate with best fitness (and its variance)
            best_var_idx        : int   = np.argmin(variances)
            best_fitness_idx    : int   = np.argmin(fitnesses)

            logger.info(f"Initial candidate with best variance: Var = {variances[best_var_idx]:.5f}, \
                                                                Fitness = {fitnesses[best_var_idx]:.5f}, \
                                                                Var Ratio = {variance_ratios[best_var_idx]:.5f}, \
                                                                Percentage Price Increase = {percentage_price_increases[best_var_idx]:.5f}%"
            )
            logger.info(f"Initial candidate with best fitness: Var = {variances[best_fitness_idx]:.5f}, \
                                                            Fitness = {fitnesses[best_fitness_idx]:.5f}\
                                                            Var Ratio = {variance_ratios[best_fitness_idx]:.5f}, \
                                                            Percentage Price Increase = {percentage_price_increases[best_fitness_idx]:.5f}%"
            )
            logger.info(f"  Time taken: {init_end_time - start_time:.1f}s")

            # Main DE Loop
            for gen in range(MAX_ITER):
                gen_start_time = time.time()
                
                # --- 1. CREATE TRIALS (Vectorized Math) ---
                # randomly select 3 candidates A, B, C ("children") for each candidate ("parent")
                # although technically A, B, C should be distinct and not equal to the parent, enforcing this is costly, so we skip this step
                # the children of all pop_size candidates are selected together in a vectorized manner, so each idxs_* has shape (pop_size,)
                idxs_a: npt.NDArray[np.int_]    = np.random.randint(low = 0, high = POP_SIZE, size = POP_SIZE)
                idxs_b: npt.NDArray[np.int_]    = np.random.randint(low = 0, high = POP_SIZE, size = POP_SIZE)
                idxs_c: npt.NDArray[np.int_]    = np.random.randint(low = 0, high = POP_SIZE, size = POP_SIZE)            

                # V = A + F * (B - C)
                # (B - C) : calculates the distance vector between B and C
                # F       : scales the distance vector (differential weight)
                # A + ... : shifts the scaled vector to start from A (the base vector)
                # mutant has shape (pop_size, optimization_dims)
                mutant: npt.NDArray[np.float64]  = population[idxs_a] + DIFF_WEIGHT * (population[idxs_b] - population[idxs_c])

                # enforce bounds
                mutant: npt.NDArray[np.float64]  = np.maximum(mutant, lower_bounds)
                mutant: npt.NDArray[np.float64]  = np.minimum(mutant, upper_bounds_final)

                # Crossover
                # for each variable, randomly decide whether to take from mutant or parent
                # if rand < cr, take from mutant; else take from parent; setting higher cr is more exploratory
                rand_mask       : npt.NDArray[np.float64] = np.random.rand(POP_SIZE, optimization_dims)
                trial_population: npt.NDArray[np.float64] = np.where(rand_mask < CROSS_PROB, mutant, population)
                

                # --- 2. EVALUATE TRIALS (PARALLEL BOTTLENECK) ---
                trial_results   : npt.NDArray[np.float64] = np.array(pool.map(evaluate_single_candidate_worker, trial_population))

                trial_fitnesses                 : npt.NDArray[np.float64] = trial_results[:,0]
                trial_variances                 : npt.NDArray[np.float64] = trial_results[:,1]
                trial_variance_ratios           : npt.NDArray[np.float64] = trial_results[:,2]
                trial_percentage_price_increases: npt.NDArray[np.float64] = trial_results[:,3]


                # --- 3. SELECTION ---
                # Replace parents with trials if trials are better
                winners: npt.NDArray[np.bool_] = trial_fitnesses < fitnesses    # Boolean mask of winners
                
                population[winners]                 = trial_population[winners]
                fitnesses[winners]                  = trial_fitnesses[winners]
                variances[winners]                  = trial_variances[winners]
                variance_ratios[winners]            = trial_variance_ratios[winners]
                percentage_price_increases[winners] = trial_percentage_price_increases[winners]
                

                # --- 4. Early Stopping ---
                # Check if any candidate meets variance threshold
                # if so, early stop by taking the candidate with the lowest fitness among them
                meet_threshold: npt.NDArray[np.bool_] = variances <= VAR_THRESHOLD
                if np.any(meet_threshold):
                    qualified_indices           : npt.NDArray[np.int_]      = np.where(meet_threshold)[0]
                    qualified_fitnesses         : npt.NDArray[np.float64]   = fitnesses[qualified_indices]
                    best_idx_within_qualified   : int                       = qualified_indices[np.argmin(qualified_fitnesses)]
                    best_vector                 : npt.NDArray[np.float64]   = population[best_idx_within_qualified].copy()

                    logger.info(f"Early stopping at generation {gen+1} with Var = {variances[best_idx_within_qualified]:.5f}, \
                                                                            Fitness = {fitnesses[best_idx_within_qualified]:.5f}, \
                                                                            Var Ratio = {variance_ratios[best_idx_within_qualified]:.5f}, \
                                                                            Percentage Price Increase = {percentage_price_increases[best_idx_within_qualified]:.5f}%"
                    )
                    logger.info(f"  Time taken: {time.time() - start_time:.1f}s")

                    return _expand_trajectory(
                        candidate_flat  = best_vector       ,
                        M               = M                 ,
                        VARS_PER_STEP   = VARS_PER_STEP     ,
                        lower_bounds_a  = lower_bounds_a    ,
                        lower_bounds_b  = lower_bounds_b    ,
                        lower_bounds_r  = lower_bounds_r    ,
                        upper_bounds_r  = upper_bounds_r    ,
                        TIMESTEPS       = TIMESTEPS       ,
                    )   
                
                # --- 5. Update Best Trackers ---
                best_var_idx        = np.argmin(variances)
                best_fitness_idx    = np.argmin(fitnesses)

                # Estimate remaining time
                gen_duration = time.time() - gen_start_time
                est_remaining_time = gen_duration * (MAX_ITER - gen - 1)

                logger.info(f"Gen {gen+1}/{MAX_ITER}")
                logger.info(f"  Current candidate with best variance: Var = {variances[best_var_idx]:.5f}, \
                                                                    Fitness = {fitnesses[best_var_idx]:.5f}, \
                                                                    Var Ratio = {variance_ratios[best_var_idx]:.5f}, \
                                                                    Percentage Price Increase = {percentage_price_increases[best_var_idx]:.5f}%"
                )
                logger.info(f"  Current candidate with best fitness: Var = {variances[best_fitness_idx]:.5f}, \
                                                                    Fitness = {fitnesses[best_fitness_idx]:.5f}, \
                                                                    Var Ratio = {variance_ratios[best_fitness_idx]:.5f}, \
                                                                    Percentage Price Increase = {percentage_price_increases[best_fitness_idx]:.5f}%"
                )
                logger.info(f"  Generation time: {gen_duration:.1f}s | Estimated remaining time: {est_remaining_time:.1f}s")


        end_time = time.time()
        logger.info(f"DE completed in {end_time - start_time:.1f}s.")
        logger.info(f"Picking candidate with best fitness...")

        best_vector : npt.NDArray[np.float64] = population[best_fitness_idx].copy()

        return _expand_trajectory(
            candidate_flat  = best_vector       ,
            M               = M                 ,
            VARS_PER_STEP   = VARS_PER_STEP     ,
            lower_bounds_a  = lower_bounds_a    ,
            lower_bounds_b  = lower_bounds_b    ,
            lower_bounds_r  = lower_bounds_r    ,
            upper_bounds_r  = upper_bounds_r    ,
            TIMESTEPS       = TIMESTEPS         ,
        )
    except Exception as e:
        raise e
    finally:
        listener.stop()
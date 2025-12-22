import numpy as np
import numpy.typing as npt
import time
from scipy.interpolate import interp1d
import multiprocessing
import os
from functools import partial

from model_leader import leader_model
from model_follower import follower_model
from networkClass import Node, Arc, ArcType
from logger import Logger


def _expand_to_full_trajectory(
        compressed_vector   : npt.NDArray[np.float_],     
        num_anchors         : int                   ,
        vars_per_step       : int                   ,
        T                   : int                   , 
        anchor_indices      : npt.NDArray[np.int_]  ,
    ) -> npt.NDArray[np.float_]:
    """
    Interpolates num_anchors back into T time steps.
    Input: compressed_vector (Shape: 1, num_anchors * vars_per_step)
    Output: full_trajectory (Shape: T, vars_per_step)
    """
    # reshape compressed vector into (num_anchors, vars_per_step)
    anchors = compressed_vector.reshape(num_anchors, vars_per_step)
    x_anchors = anchor_indices
    x_target = np.arange(T)
    
    # Linear interpolation is safe and robust
    f = interp1d(x_anchors, anchors, axis=0, kind='linear', fill_value="extrapolate")
    return f(x_target) # Returns shape (T, vars_per_step)


def _reference_candidate(
        candidate_flat,
        **kwargs
    ):
    """
    Generate the reference variance value by solving the follower problem with minimum prices.
    """
    # ----------------------------
    # Parameters
    # ----------------------------
    # Follower model parameters
    N                   : int                               = kwargs.get("N")                       # number of operation zones (1, ..., N)
    T                   : int                               = kwargs.get("T")                       # termination time of daily operations (0, ..., T)
    L                   : int                               = kwargs.get("L")                       # max SoC level (all EVs start at this level) (0, ..., L)
    W                   : int                               = kwargs.get("W")                       # maximum time intervals a passenger will wait for a ride (0, ..., W-1; demand expires at W)
    travel_demand       : dict[tuple[int, int, int], int]   = kwargs.get("travel_demand")           # travel demand from zone i to j at starting at time t
    travel_time         : dict[tuple[int, int, int], int]   = kwargs.get("travel_time")             # travel time from i to j at starting at time t
    travel_energy       : dict[tuple[int, int], int]        = kwargs.get("travel_energy")           # energy consumed for trip from zone i to j
    order_revenue       : dict[tuple[int, int, int], float] = kwargs.get("order_revenue")           # order revenue for each trip served from i to j at time t
    penalty             : dict[tuple[int, int, int], float] = kwargs.get("penalty")                 # penalty cost for each unserved trip from i to j at time t 
    L_min               : int                               = kwargs.get("L_min")                   # min SoC level all EV must end with at the end of the daily operations
    num_EVs             : int                               = kwargs.get("num_EVs")                 # total number of EVs in the fleet
    num_ports           : dict[int, int]                    = kwargs.get("num_ports")               # number of chargers in each zone
    elec_supplied       : dict[tuple[int, int], int]        = kwargs.get("elec_supplied")           # electricity supplied (in SoC levels) at zone i at time t
    max_charge_speed    : int                               = kwargs.get("max_charge_speed")        # max charging speed (in SoC levels) of one EV in one time step

    charge_cost_low     : dict[int, float]                  = kwargs.get("charge_cost_low")         # a_t
    charge_cost_high    : dict[int, float]                  = kwargs.get("charge_cost_high")        # b_t
    elec_threshold      : dict[int, int]                    = kwargs.get("elec_threshold")          # r_t

    # DE parameters
    num_anchors         : int                               = kwargs.get("num_anchors")             # number of anchors for DE
    vars_per_step       : int                               = kwargs.get("vars_per_step")           # number of variables per time step (3: a_t, b_t, r_t)
    anchor_indices      : npt.NDArray[np.int_]              = kwargs.get("anchor_indices")          # indices of the anchors in the full T time steps


    # ----------------------------
    # Solve Follower Problem
    # ----------------------------

    # Expand 12 params -> 144 params
    full_trajectory: npt.NDArray[np.float_] = _expand_to_full_trajectory(
        candidate_flat,
        num_anchors     = num_anchors         ,
        vars_per_step   = vars_per_step       ,
        T               = T                   ,
        anchor_indices  = anchor_indices      ,
    )
    
    # charge_cost_low (a_t): first column
    # charge_cost_high (b_t): second column
    # elec_threshold (r_t): third column
    charge_cost_low     : dict[int, float]                  = {t: full_trajectory[t, 0] for t in range(T + 1)}  # a_t
    charge_cost_high    : dict[int, float]                  = {t: full_trajectory[t, 1] for t in range(T + 1)}  # b_t
    elec_threshold      : dict[int, int]                    = {t: full_trajectory[t, 2] for t in range(T + 1)}  # r_t

    follower_outputs = follower_model(
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

        charge_cost_low         = charge_cost_low       ,
        charge_cost_high        = charge_cost_high      ,
        elec_threshold          = elec_threshold        ,

        relaxed                 = True                  ,

        to_console              = False                 ,
        to_file                 = False                 ,
    )

    # Extract variables and sets from the follower_outputs    
    x               : dict[int, float]      = follower_outputs.get("sol", {}).get("x", {})
    all_arcs        : dict[int, Arc]        = follower_outputs.get("arcs", {}).get("all_arcs", {})
    charge_arcs_t   : dict[int, set[int]]   = follower_outputs.get("arcs", {}).get("charge_arcs_t", {})

    # Calculate electricity consumption at each time step using vectorized operations
    electricity_usage: npt.NDArray[np.float_] = np.zeros(T + 1)

    for t in range(1, T):  # Exclude time 0 and T
        # Calculate total electricity used at time t
        for e_id in charge_arcs_t.get(t, set()):
            arc = all_arcs[e_id]
            
            # Electricity used = number of EVs * charge amount
            charge_amount = arc.d.l - arc.o.l  # SoC levels charged
            electricity_usage[t] += x[e_id] * charge_amount

    # Calculate variance of electricity consumption using numpy
    usage_vector    : npt.NDArray[np.float_] = electricity_usage[1:T]  # exclude time 0 and T
    variance        : float                  = np.var(usage_vector, ddof=1) if len(usage_vector) > 1 else 0.0

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
    N                   : int                               = kwargs.get("N")                       # number of operation zones (1, ..., N)
    T                   : int                               = kwargs.get("T")                       # termination time of daily operations (0, ..., T)
    L                   : int                               = kwargs.get("L")                       # max SoC level (all EVs start at this level) (0, ..., L)
    W                   : int                               = kwargs.get("W")                       # maximum time intervals a passenger will wait for a ride (0, ..., W-1; demand expires at W)
    travel_demand       : dict[tuple[int, int, int], int]   = kwargs.get("travel_demand")           # travel demand from zone i to j at starting at time t
    travel_time         : dict[tuple[int, int, int], int]   = kwargs.get("travel_time")             # travel time from i to j at starting at time t
    travel_energy       : dict[tuple[int, int], int]        = kwargs.get("travel_energy")           # energy consumed for trip from zone i to j
    order_revenue       : dict[tuple[int, int, int], float] = kwargs.get("order_revenue")           # order revenue for each trip served from i to j at time t
    penalty             : dict[tuple[int, int, int], float] = kwargs.get("penalty")                 # penalty cost for each unserved trip from i to j at time t 
    L_min               : int                               = kwargs.get("L_min")                   # min SoC level all EV must end with at the end of the daily operations
    num_EVs             : int                               = kwargs.get("num_EVs")                 # total number of EVs in the fleet
    num_ports           : dict[int, int]                    = kwargs.get("num_ports")               # number of chargers in each zone
    elec_supplied       : dict[tuple[int, int], int]        = kwargs.get("elec_supplied")           # electricity supplied (in SoC levels) at zone i at time t
    max_charge_speed    : int                               = kwargs.get("max_charge_speed")        # max charging speed (in SoC levels) of one EV in one time step
    
    # Leader model parameters
    penalty_weight      : float                             = kwargs.get("penalty_weight")          # penalty weight for high a_t and b_t
    wholesale_elec_price: dict[int, float]                  = kwargs.get("wholesale_elec_price")    # wholesale electricity price at time t
    reference_variance  : float                             = kwargs.get("reference_variance")      # reference variance for normalization
    
    # DE parameters
    num_anchors         : int                               = kwargs.get("num_anchors")             # number of anchors for DE
    vars_per_step       : int                               = kwargs.get("vars_per_step")           # number of variables per time step (3: a_t, b_t, r_t)
    anchor_indices      : npt.NDArray[np.int_]              = kwargs.get("anchor_indices")          # indices of the anchors in the full T time steps


    # Expand 12 params -> 144 params
    full_trajectory: npt.NDArray[np.float_] = _expand_to_full_trajectory(
        candidate_flat,
        num_anchors     = num_anchors         ,
        vars_per_step   = vars_per_step       ,
        T               = T                   ,
        anchor_indices  = anchor_indices      ,
    )
    
    # charge_cost_low (a_t): first column
    # charge_cost_high (b_t): second column
    # elec_threshold (r_t): third column
    charge_cost_low     : dict[int, float]                  = {t: full_trajectory[t, 0] for t in range(T + 1)}  # a_t
    charge_cost_high    : dict[int, float]                  = {t: full_trajectory[t, 1] for t in range(T + 1)}  # b_t
    elec_threshold      : dict[int, int]                    = {t: full_trajectory[t, 2] for t in range(T + 1)}  # r_t


    # ----------------------------
    # Solve Follower Problem
    # ----------------------------
    follower_outputs = follower_model(
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

        charge_cost_low         = charge_cost_low       ,
        charge_cost_high        = charge_cost_high      ,
        elec_threshold          = elec_threshold        ,

        relaxed                 = True                  ,

        to_console              = False                 ,
        to_file                 = False                 ,
    )

    # Extract variables and sets from the follower_outputs    
    x               : dict[int, float]      = follower_outputs.get("sol", {}).get("x", {})

    all_arcs        : dict[int, Arc]        = follower_outputs.get("arcs", {}).get("all_arcs", {})
    charge_arcs_t   : dict[int, set[int]]   = follower_outputs.get("arcs", {}).get("charge_arcs_t", {})

    ZONES           : list[int]             = follower_outputs.get("sets", {}).get("ZONES", [])


    # ----------------------------
    # Solve Leader Problem
    # ----------------------------
    fitness, variance, variance_ratio, percentage_price_increase = leader_model(
        # Follower model parameters
        T                       = T                     ,
        elec_supplied           = elec_supplied         ,

        charge_cost_low         = charge_cost_low       ,
        charge_cost_high        = charge_cost_high      ,
        elec_threshold          = elec_threshold        ,

        # Leader model parameters
        penalty_weight          = penalty_weight        ,
        wholesale_elec_price    = wholesale_elec_price  ,
        reference_variance      = reference_variance    ,    

        # Follower solution variables
        x                       = x                     ,
        
        # Arcs
        all_arcs                = all_arcs              ,
        charge_arcs_t           = charge_arcs_t         ,

        # Sets
        ZONES                   = ZONES                 ,

        # Metadata
        to_console              = False                 ,
        to_file                 = False                 ,
    )

    return fitness, variance, variance_ratio, percentage_price_increase


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
    N                   : int                               = kwargs.get("N")                       # number of operation zones (1, ..., N)
    T                   : int                               = kwargs.get("T")                       # termination time of daily operations (0, ..., T)
    L                   : int                               = kwargs.get("L")                       # max SoC level (all EVs start at this level) (0, ..., L)
    W                   : int                               = kwargs.get("W")                       # maximum time intervals a passenger will wait for a ride (0, ..., W-1; demand expires at W)
    travel_demand       : dict[tuple[int, int, int], int]   = kwargs.get("travel_demand")           # travel demand from zone i to j at starting at time t
    travel_time         : dict[tuple[int, int, int], int]   = kwargs.get("travel_time")             # travel time from i to j at starting at time t
    travel_energy       : dict[tuple[int, int], int]        = kwargs.get("travel_energy")           # energy consumed for trip from zone i to j
    order_revenue       : dict[tuple[int, int, int], float] = kwargs.get("order_revenue")           # order revenue for each trip served from i to j at time t
    penalty             : dict[tuple[int, int, int], float] = kwargs.get("penalty")                 # penalty cost for each unserved trip from i to j at time t 
    L_min               : int                               = kwargs.get("L_min")                   # min SoC level all EV must end with at the end of the daily operations
    num_EVs             : int                               = kwargs.get("num_EVs")                 # total number of EVs in the fleet  
    num_ports           : dict[int, int]                    = kwargs.get("num_ports")               # number of chargers in each zone
    elec_supplied       : dict[tuple[int, int], int]        = kwargs.get("elec_supplied")           # electricity supplied (in SoC levels) at zone i at time t
    max_charge_speed    : int                               = kwargs.get("max_charge_speed")        # max charging speed (in SoC levels) of one EV in one time step
    
    # Leader model parameters
    wholesale_elec_price: dict[int, float]                  = kwargs.get("wholesale_elec_price")    # wholesale electricity price at time t
    penalty_weight      : float                             = kwargs.get("penalty_weight")          # penalty weight for high a_t and b_t
    
    # DE parameters
    pop_size            : int                               = kwargs.get("pop_size")                # population size for DE
    max_iter            : int                               = kwargs.get("max_iter")                # max iterations for DE
    f                   : float                             = kwargs.get("f")                       # differential weight
    cr                  : float                             = kwargs.get("cr")                      # crossover probability
    var_threshold       : float                             = kwargs.get("var_threshold")           # variance threshold for stopping criteria
    num_anchors         : int                               = kwargs.get("num_anchors")             # number of anchors for DE
    dims_per_step       : int                               = kwargs.get("dims_per_step")           # number of variables per time step (3: a_t, b_t, r_t)

    # Metadata
    timestamp           : str                               = kwargs.get("timestamp", "")           # timestamp for logging
    file_name           : str                               = kwargs.get("file_name", "")           # filename for logging
    folder_name         : str                               = kwargs.get("folder_name", "")         # folder name for logging


    logger = Logger("bilevel_DE", level="DEBUG", to_console=False, timestamp=timestamp)
    logger.save (os.path.join (folder_name, f"bilevel_DE_{file_name}"))
    logger.info("Parameters loaded successfully")


    # ----------------------------
    # DE Setup
    # ----------------------------

    # Using numpy array is much faster for numerical operations
    wholesale_elec_price_arr: npt.NDArray[np.float_] = np.array([wholesale_elec_price[t] for t in range(T)])

    # Lower bounds
    lower_bounds_a          : npt.NDArray[np.float_] = wholesale_elec_price_arr.copy()          # a_t >= wholesale price
    lower_bounds_b          : npt.NDArray[np.float_] = np.zeros(T)                              # b_t >= 0
    lower_bounds_r          : npt.NDArray[np.float_] = np.zeros(T)                              # r_t >= 0

    # Upper bounds
    upper_bounds_a_init     : npt.NDArray[np.float_] = wholesale_elec_price_arr * 2.0                                                   # a_t <= 2 * wholesale price (initial, can be exceeded during evolution)
    upper_bounds_b_init     : npt.NDArray[np.float_] = wholesale_elec_price_arr * 1.0                                                   # b_t <= wholesale price (initial, can be exceeded during evolution)
    upper_bounds_r          : npt.NDArray[np.float_] = np.array([sum(elec_supplied.get((i,t),0) for i in range(N)) for t in range(T)])  # r_t <= total electricity supplied at time t (hard limit)

    # a_t and b_t can be unbounded in theory, but we set a very high upper bound for numerical stability
    upper_bounds_a_final    : npt.NDArray[np.float_] = wholesale_elec_price_arr * 10.0
    upper_bounds_b_final    : npt.NDArray[np.float_] = wholesale_elec_price_arr * 10.0

    # anchor indices (dimensionality reduction)
    # instead of optimizing for all T time steps, we pick num_anchors key points and interpolate the rest
    anchor_indices          : npt.NDArray[np.int_]   = np.linspace(0, T - 1, num_anchors, dtype=int)
    optimization_dims       : int                    = num_anchors * dims_per_step

    # array of bounds for anchored variables only
    # Flattened in the order of [a_0, b_0, r_0, a_1, b_1, r_1, ..., a_T-1, b_T-1, r_T-1]
    lower_bounds            : npt.NDArray[np.float_] = np.column_stack(
        (lower_bounds_a, lower_bounds_b, lower_bounds_r)
    )[anchor_indices].flatten()

    upper_bounds_init       : npt.NDArray[np.float_] = np.column_stack(
        (upper_bounds_a_init, upper_bounds_b_init, upper_bounds_r)
    )[anchor_indices].flatten()

    upper_bounds_final      : npt.NDArray[np.float_] = np.column_stack(
        (upper_bounds_a_final, upper_bounds_b_final, upper_bounds_r)
    )[anchor_indices].flatten()

    # Initialize Population
    # each row is a candidate represented as a 1D array of length optimization_dims
    # e.g. [a_0, b_0, r_0, a_1, b_1, r_1, ..., a_T-1, b_T-1, r_T-1]
    # number of rows = pop_size
    population              : npt.NDArray[np.float_] = np.random.uniform (
        low     = lower_bounds, 
        high    = upper_bounds_init, 
        size    = (pop_size, optimization_dims)
    )
    logger.info(f"DE population initialized with size {population.shape}")

    # Ensure that population contain one candidate with all variables at lower bounds
    # This ensures that setting minimum prices is always considered
    population[0,:] = lower_bounds.copy()


    # ----------------------------
    # DE Optimization
    # ----------------------------
    
    # Get reference variance using the candidate with minimum prices
    reference_candidate = lower_bounds.copy()
    reference_variance = _reference_candidate(
        reference_candidate,
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

        # DE parameters
        num_anchors             = num_anchors           ,
        vars_per_step           = dims_per_step         ,
        anchor_indices          = anchor_indices        ,
    )
    logger.info(f"Reference variance obtained: {reference_variance:.5f}")

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

        # Leader model parameters
        penalty_weight          = penalty_weight        ,
        wholesale_elec_price    = wholesale_elec_price  ,    
        reference_variance      = reference_variance    ,

        # DE parameters
        num_anchors             = num_anchors           ,
        vars_per_step           = dims_per_step         ,
        anchor_indices          = anchor_indices        ,
    )

    start_time = time.time()

    # Initial Evaluation
    # We use a Pool to run all candidates in parallel
    logger.info(f"Initializing pool with {multiprocessing.cpu_count()} cores...")

    with multiprocessing.Pool() as pool:

        # Calculate initial fitness (Parallel)
        # Runs the evaluate_single_candidate function on each candidate in the population in parallel
        results     : npt.NDArray[np.float_]    = np.array(pool.map(evaluate_single_candidate_worker, population))

        # Extract fitness and variance (first and second columns of each row)
        fitnesses                   : npt.NDArray[np.float_]    = results[:,0]
        variances                   : npt.NDArray[np.float_]    = results[:,1]
        variance_ratios             : npt.NDArray[np.float_]    = results[:,2]
        percentage_price_increases  : npt.NDArray[np.float_]    = results[:,3]

        init_end_time       : float = time.time()

        # Check if any candidate meets variance threshold
        # if so, early stop by taking the candidate with the lowest fitness among them
        meet_threshold: npt.NDArray[np.bool_] = variances <= var_threshold

        if np.any(meet_threshold):
            qualified_indices           : npt.NDArray[np.int_]      = np.where(meet_threshold)[0]
            qualified_fitnesses         : npt.NDArray[np.float_]    = fitnesses[qualified_indices]
            best_idx_within_qualified   : int                       = qualified_indices[np.argmin(qualified_fitnesses)]
            best_vector                 : npt.NDArray[np.float_]    = population[best_idx_within_qualified].copy()

            logger.info(f"Early stopping at initialization with Var = {variances[best_idx_within_qualified]:.5f}, \
                                                                Fitness = {fitnesses[best_idx_within_qualified]:.5f}, \
                                                                Var Ratio =  {variance_ratios[best_idx_within_qualified]:.5f}, \
                                                                Percentage Price Increase = {percentage_price_increases[best_idx_within_qualified]:.5f}%"
            )
            logger.info(f"  Time taken: {init_end_time - start_time:.1f}s")

            return _expand_to_full_trajectory(
                best_vector,
                num_anchors     = num_anchors,
                vars_per_step   = dims_per_step,
                T               = T,
                anchor_indices  = anchor_indices,
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
        for gen in range(max_iter):
            gen_start_time = time.time()
            
            # --- 1. CREATE TRIALS (Vectorized Math) ---
            # randomly select 3 candidates A, B, C ("children") for each candidate ("parent")
            # although technically A, B, C should be distinct and not equal to the parent, enforcing this is costly, so we skip this step
            # the children of all pop_size candidates are selected together in a vectorized manner, so each idxs_* has shape (pop_size,)
            idxs_a: npt.NDArray[np.int_]    = np.random.randint(low = 0, high = pop_size, size = pop_size)
            idxs_b: npt.NDArray[np.int_]    = np.random.randint(low = 0, high = pop_size, size = pop_size)
            idxs_c: npt.NDArray[np.int_]    = np.random.randint(low = 0, high = pop_size, size = pop_size)            

            # V = A + F * (B - C)
            # (B - C) : calculates the distance vector between B and C
            # F       : scales the distance vector (differential weight)
            # A + ... : shifts the scaled vector to start from A (the base vector)
            # mutant has shape (pop_size, optimization_dims)
            mutant: npt.NDArray[np.float_]  = population[idxs_a] + f * (population[idxs_b] - population[idxs_c])

            # enforce bounds
            mutant: npt.NDArray[np.float_]  = np.maximum(mutant, lower_bounds)
            mutant: npt.NDArray[np.float_]  = np.minimum(mutant, upper_bounds_final)

            # Ensure that threshold dimensions (r_t) are integers
            for anchor_idx in range(num_anchors):
                r_dim_idx = anchor_idx * dims_per_step + 2  # r_t is the 3rd variable in each time step
                mutant[:, r_dim_idx] = np.round(mutant[:, r_dim_idx])

            # Crossover
            # for each variable, randomly decide whether to take from mutant or parent
            # if rand < cr, take from mutant; else take from parent; setting higher cr is more exploratory
            rand_mask       : npt.NDArray[np.float_] = np.random.rand(pop_size, optimization_dims)
            trial_population: npt.NDArray[np.float_] = np.where(rand_mask < cr, mutant, population)
            

            # --- 2. EVALUATE TRIALS (PARALLEL BOTTLENECK) ---
            trial_results   : npt.NDArray[np.float_] = np.array(pool.map(evaluate_single_candidate_worker, trial_population))

            trial_fitnesses                 : npt.NDArray[np.float_] = trial_results[:,0]
            trial_variances                 : npt.NDArray[np.float_] = trial_results[:,1]
            trial_variance_ratios           : npt.NDArray[np.float_] = trial_results[:,2]
            trial_percentage_price_increases: npt.NDArray[np.float_] = trial_results[:,3]


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
            meet_threshold: npt.NDArray[np.bool_] = variances <= var_threshold
            if np.any(meet_threshold):
                qualified_indices           : npt.NDArray[np.int_]      = np.where(meet_threshold)[0]
                qualified_fitnesses         : npt.NDArray[np.float_]    = fitnesses[qualified_indices]
                best_idx_within_qualified   : int                       = qualified_indices[np.argmin(qualified_fitnesses)]
                best_vector                 : npt.NDArray[np.float_]    = population[best_idx_within_qualified].copy()

                logger.info(f"Early stopping at generation {gen+1} with Var = {variances[best_idx_within_qualified]:.5f}, \
                                                                        Fitness = {fitnesses[best_idx_within_qualified]:.5f}, \
                                                                        Var Ratio = {variance_ratios[best_idx_within_qualified]:.5f}, \
                                                                        Percentage Price Increase = {percentage_price_increases[best_idx_within_qualified]:.5f}%"
                )
                logger.info(f"  Time taken: {time.time() - start_time:.1f}s")

                return _expand_to_full_trajectory(
                    best_vector,
                    num_anchors     = num_anchors,
                    vars_per_step   = dims_per_step,
                    T               = T,
                    anchor_indices  = anchor_indices,
                )
            
            # --- 5. Update Best Trackers ---
            best_var_idx        = np.argmin(variances)
            best_fitness_idx    = np.argmin(fitnesses)

            # Estimate remaining time
            gen_duration = time.time() - gen_start_time
            est_remaining_time = gen_duration * (max_iter - gen - 1)

            logger.info(f"Gen {gen+1}/{max_iter}")
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

    best_vector : npt.NDArray[np.float_] = population[best_fitness_idx].copy()

    # Expand 12 params -> 144 params
    full_trajectory: npt.NDArray[np.float_] = _expand_to_full_trajectory(
        best_vector,
        num_anchors     = num_anchors         ,
        vars_per_step   = dims_per_step       ,
        T               = T                   ,
        anchor_indices  = anchor_indices      ,
    )
    
    # charge_cost_low (a_t): first column
    # charge_cost_high (b_t): second column
    # elec_threshold (r_t): third column
    charge_cost_low     : dict[int, float]                  = {t: full_trajectory[t, 0] for t in range(T + 1)}  # a_t
    charge_cost_high    : dict[int, float]                  = {t: full_trajectory[t, 1] for t in range(T + 1)}  # b_t
    elec_threshold      : dict[int, int]                    = {t: full_trajectory[t, 2] for t in range(T + 1)}  # r_t

    return {
        "charge_cost_low"    : charge_cost_low        ,
        "charge_cost_high"   : charge_cost_high       ,
        "elec_threshold"     : elec_threshold         ,
    }

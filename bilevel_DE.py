import numpy as np
import numpy.typing as npt
import time
import multiprocessing
import os
from functools import partial

from model_leader import leader_model
from model_follower import follower_model
from networkClass import Arc
from logger import Logger, LogListener
from exceptions import OptimizationError
from config_DE import (
    POP_SIZE        ,
    NUM_CORES       ,
    MAX_ITER        ,
    DIFF_WEIGHT     ,
    CROSS_PROB      ,
    VAR_THRESHOLD   ,
    NUM_ANCHORS     ,
    VARS_PER_STEP   ,
)

def _precompute_interpolation_matrix(
        T               : int                   , 
        anchor_indices  : npt.NDArray[np.int_]  ,
    ) -> npt.NDArray[np.float64]:
    """
    Creates a (T-1, num_anchors) interpolation matrix M such that: Interior_Trajectory = M @ Anchors <br>
    Where: \n
        - Interior_Trajectory: (T-1, 3) matrix of interpolated values for time steps 1 to T-1
        - Anchors: (num_anchors, 3) matrix of anchor point values

    The rows of M correspond to time steps 1 to T-1, and the columns correspond to anchor points. <br>
    The values in M are the weights for linear interpolation (each row sums to 1). <br>
    For example, for T=5 and num_anchors=3 at indices [1, 2, 4]: <br>
    ```
        | 1.0   0.0    0.0  |  # t=1 
        | 0.0   1.0    0.0  |  # t=2 
        | 0.0   0.5    0.5  |  # t=3 
        | 0.0   0.0    1.0  |  # t=4 
    ```
    Note that we only interpolates time steps 1 to T-1 (excludes time steps 0 and T)
    """
    M: npt.NDArray[np.float64] = np.zeros((T-1, NUM_ANCHORS))
    
    for i in range(NUM_ANCHORS - 1):

        anchor_start    = anchor_indices[i]         # eg 1
        anchor_end      = anchor_indices[i+1]       # eg 3
        segment_len     = anchor_end - anchor_start # eg 3 - 1 = 2

        if segment_len == 0:
            raise ValueError("Duplicate anchors detected, segment length is zero")
            
        slope = 1.0 / segment_len
        
        # Fill the matrix rows corresponding to this time segment
        # Note: rows are indexed from 0 to T-2, representing time steps 1 to T-1
        for t in range(anchor_start, anchor_end + 1): # +1 to include end for continuity
            if t < 1 or t >= T:  # Skip time steps 0 and T
                raise ValueError("Time step t out of interpolation range")
            local_x = t - anchor_start
            weight_next = local_x * slope
            weight_prev = 1.0 - weight_next
            
            # M[t-1, i] is weight of anchor i (t-1 because we skip time step 0)
            # M[t-1, i+1] is weight of anchor i+1
            M[t-1, i] = weight_prev
            M[t-1, i+1] = weight_next
            
    return M


def _expand_trajectory (
        candidate_flat  : npt.NDArray[np.float64]   , 
        M               : npt.NDArray[np.float64]   , 
        T               : int                       ,
        lower_bounds_a  : npt.NDArray[np.float64]   ,
        lower_bounds_b  : npt.NDArray[np.float64]   ,
        lower_bounds_r  : npt.NDArray[np.float64]   ,
        upper_bounds_r  : npt.NDArray[np.float64]   ,
        TIMESTEPS       : list[int]                 ,
    ) -> dict[str, dict[int, float]]:
    """
    Expands the compressed candidate solution into full pricing trajectories for a_t, b_t, r_t with bounds enforcement. <br>
    Input: \n
        - candidate_flat: (num_anchors * vars_per_step) 1D array flattened of candidate solution
    Output: \n
        - Dictionary containing full trajectories for charge_cost_low (a_t), charge_cost_high (b_t), elec_threshold (r_t)
        - Each trajectory is a dict mapping time step t to its value for all time steps 0 to T
    Note that the interpolation only applies to time steps 1 to T-1; time steps 0 and T are set to their lower bounds. <br>
    """
    # compressed shape: (num_anchors * vars_per_step)
    # reshape to (num_anchors, vars_per_step)
    anchors = candidate_flat.reshape(-1, VARS_PER_STEP)
    
    # (T-1, num_anchors) @ (num_anchors, 3) -> (T-1, 3)
    # This gives us interpolated values for time steps 1 to T-1
    interior_trajectory = M @ anchors

    # Build full trajectory with fixed values at time steps 0 and T
    full_trajectory = np.zeros((T+1, VARS_PER_STEP))
    
    # Time step 0: use lower bounds
    full_trajectory[0, 0] = lower_bounds_a[0]
    full_trajectory[0, 1] = lower_bounds_b[0]
    full_trajectory[0, 2] = lower_bounds_r[0]
    
    # Time steps 1 to T-1: use interpolated values with bounds enforcement
    full_trajectory[1:T, 0] = np.maximum(interior_trajectory[:, 0], lower_bounds_a[1:T])  # a_t
    full_trajectory[1:T, 1] = np.maximum(interior_trajectory[:, 1], lower_bounds_b[1:T])  # b_t
    full_trajectory[1:T, 2] = np.clip(interior_trajectory[:, 2], lower_bounds_r[1:T], upper_bounds_r[1:T])  # r_t
    
    # Time step T: use lower bounds
    full_trajectory[T, 0] = lower_bounds_a[T]
    full_trajectory[T, 1] = lower_bounds_b[T]
    full_trajectory[T, 2] = lower_bounds_r[T]

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
    T                       : int                                   = kwargs["T"]                       # termination time of daily operations (0, ..., T)

    # Network components
    all_arcs                : dict[int                  , Arc]      = kwargs["all_arcs"]
    charge_arcs_t           : dict[int                  , set[int]] = kwargs["charge_arcs_t"]
    TIMESTEPS               : list[int]                             = kwargs["TIMESTEPS"]

    # Path Metadata
    timestamp               : str                                   = kwargs["timestamp"]               # timestamp for logging
    file_name               : str                                   = kwargs["file_name"]               # filename for logging
    folder_name             : str                                   = kwargs["folder_name"]             # folder name for logging

    # Metadata
    log_queue               : multiprocessing.Queue                 = kwargs["log_queue"]               # multiprocessing log queue
    log_queue_gurobi        : multiprocessing.Queue                 = kwargs["log_queue_gurobi"]        # multiprocessing log queue for gurobi

    logger = Logger (
        f"Reference"     ,    
        level       = "DEBUG"       ,
        to_console  = False         ,
        folder_name = folder_name   ,
        file_name   = file_name     ,
        timestamp   = timestamp     ,
        queue       = log_queue     ,
    )
    logger_gurobi = Logger (
        f"Gurobi_Reference"                 ,
        level       = "DEBUG"               ,
        to_console  = False                 ,
        folder_name = folder_name           ,
        file_name   = file_name             ,
        timestamp   = timestamp             ,
        queue       = log_queue_gurobi      ,
    )

    logger.info("Solving follower problem for reference candidate with minimum prices...")

    # ----------------------------
    # Solve Follower Problem
    # ----------------------------
    try:
        follower_outputs = follower_model(
            **kwargs,
            # Metadata
            logger                  = logger                ,
            logger_gurobi           = logger_gurobi         ,
        )
    except OptimizationError as e:
        raise OptimizationError("Failed to solve follower problem for reference candidate.", details=e) from e
    except Exception as e:
        raise Exception("Unexpected error when solving follower problem for reference candidate.") from e

    logger.info("Reference candidate solved successfully.")

    # Extract variables and sets from the follower_outputs    
    x               : dict[int, float]                          = follower_outputs["x"]

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


def _evaluate_candidate(
        candidate_flat,
        **kwargs
    ):
    """
    Calculate and returns the penalty and variance for a single candidate solution.
    """    
    # ----------------------------
    # Parameters
    # ----------------------------
    # Network components
    TIMESTEPS               : list[int]                             = kwargs["TIMESTEPS"]

    # Pricing Variable bounds
    lower_bounds_a          : npt.NDArray[np.float64]               = kwargs["lower_bounds_a"]          # lower bounds for a_t
    lower_bounds_b          : npt.NDArray[np.float64]               = kwargs["lower_bounds_b"]          # lower bounds for b_t
    lower_bounds_r          : npt.NDArray[np.float64]               = kwargs["lower_bounds_r"]          # lower bounds for r_t
    upper_bounds_r          : npt.NDArray[np.float64]               = kwargs["upper_bounds_r"]          # upper bounds for r_t
    
    # Path Metadata
    timestamp               : str                                   = kwargs["timestamp"]               # timestamp for logging
    file_name               : str                                   = kwargs["file_name"]               # filename for logging
    folder_name             : str                                   = kwargs["folder_name"]             # folder name for logging

    # Metadata
    M                       : npt.NDArray[np.float64]               = kwargs["M"]                       # interpolation matrix
    log_queue               : multiprocessing.Queue                 = kwargs["log_queue"]               # multiprocessing log queue
    log_queue_gurobi        : multiprocessing.Queue                 = kwargs["log_queue_gurobi"]        # multiprocessing log queue for gurobi

    logger_worker = Logger (
        f"Worker_{os.getpid()}"     ,    
        level       = "DEBUG"       ,
        to_console  = False         ,
        folder_name = folder_name   ,
        file_name   = file_name     ,
        timestamp   = timestamp     ,
        queue       = log_queue     ,
    )

    logger_worker_gurobi = Logger (
        f"Gurobi_Worker_{os.getpid()}"      ,
        level       = "DEBUG"               ,
        to_console  = False                 ,
        folder_name = folder_name           ,
        file_name   = file_name             ,
        timestamp   = timestamp             ,
        queue       = log_queue_gurobi      ,
    )

    T = kwargs["T"]
    
    charging_price_parameters = _expand_trajectory(
        candidate_flat  = candidate_flat    ,
        M               = M                 ,
        T               = T                 ,
        lower_bounds_a  = lower_bounds_a    ,
        lower_bounds_b  = lower_bounds_b    ,
        lower_bounds_r  = lower_bounds_r    ,
        upper_bounds_r  = upper_bounds_r    ,   
        TIMESTEPS       = TIMESTEPS         ,
    )

    logger_worker.info("Solving follower problem for candidate...")

    # ----------------------------
    # Solve Follower Problem
    # ----------------------------
    try:
        solutions_candidate = follower_model(
            **kwargs                                ,
            **charging_price_parameters             ,
            # Metadata
            logger          = logger_worker         ,
            logger_gurobi   = logger_worker_gurobi  ,
        )
    except OptimizationError as e:
        raise OptimizationError("Failed to solve follower problem for candidate.", details=e) from e
    except Exception as e:
        raise Exception("Unexpected error when solving follower problem for candidate.") from e

    logger_worker.info("Follower problem for candidate solved successfully. Solving leader problem...")


    # ----------------------------
    # Solve Leader Problem
    # ----------------------------
    leader_outputs = leader_model(
        **kwargs                    ,
        **charging_price_parameters ,
        **solutions_candidate       ,
        # Metadata
        logger = logger_worker      ,
    )

    logger_worker.info("Leader problem for candidate solved successfully.")

    return (
        leader_outputs["fitness"]                   ,
        leader_outputs["variance"]                  ,
        leader_outputs["variance_ratio"]            ,
        leader_outputs["percentage_price_increase"] ,
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
    T                       : int                                   = kwargs["T"]                       # termination time of daily operations (0, ..., T)
    elec_supplied           : dict[tuple[int, int]      , int]      = kwargs["elec_supplied"]           # electricity supplied (in SoC levels) at zone i at time t

    # Network components
    ZONES                   : list[int]                             = kwargs["ZONES"]
    TIMESTEPS               : list[int]                             = kwargs["TIMESTEPS"]

    # Leader model parameters
    wholesale_elec_price    : dict[int                  , float]    = kwargs["wholesale_elec_price"]    # wholesale electricity price at time t
    
    # Path Metadata
    timestamp               : str                                   = kwargs["timestamp"]               # timestamp for logging
    file_name               : str                                   = kwargs["file_name"]               # filename for logging
    folder_name             : str                                   = kwargs["folder_name"]             # folder name for logging

    # ----------------------------
    # Logger Setup
    # ----------------------------
    # 1. Create the Queue using Manager (Safest for Pools)
    manager = multiprocessing.Manager()
    log_queue = manager.Queue()

    manager_gurobi = multiprocessing.Manager()
    log_queue_gurobi = manager_gurobi.Queue()

    with LogListener(
            "stage1_MP_evaluate_candidate",
            folder_name         , 
            file_name           , 
            timestamp           ,
            log_queue           ,
            to_console = False  ,
        ) as listener, \
        LogListener(
            "stage1_MP_gurobi_logs",
            folder_name         ,
            file_name           ,
            timestamp           ,
            log_queue_gurobi    ,
            to_console = False  ,
        ) as listener_gurobi:
        
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
        logger.info("Parameters loaded successfully!")


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

        logger.info("Bounds for decision variables set successfully!")

        # anchor indices (dimensionality reduction)
        # Only optimize time steps 1 to T-1 (exclude 0 and T)
        # Time steps 0 and T are fixed at their lower bounds
        if NUM_ANCHORS < 1:
            raise ValueError("NUM_ANCHORS must be at least 1")
        
        # Distribute anchors only within time steps 1 to T-1
        anchor_indices      : npt.NDArray[np.int_]      = np.linspace(1, T-1, NUM_ANCHORS, dtype=int)
        optimization_dims   : int                       = NUM_ANCHORS * VARS_PER_STEP
        M                   : npt.NDArray[np.float64]   = _precompute_interpolation_matrix(T, anchor_indices)

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

        logger.info("Calculating reference variance from candidate with lowest prices...")
        start_time_ref = time.time()

        try:
            reference_variance = _reference_candidate(
                charge_cost_low     = lowest_charge_cost_low    ,
                charge_cost_high    = lowest_charge_cost_high   ,
                elec_threshold      = lowest_elec_threshold     ,
                **kwargs                                        ,  
                # Metadata
                log_queue           = log_queue                 ,
                log_queue_gurobi    = log_queue_gurobi          ,
            )
        except OptimizationError as e:
            logger.error("Failed to obtain reference variance from reference candidate.")
            raise e
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise e

        end_time_ref = time.time()
        logger.info(f"Reference variance obtained: {reference_variance:.5f} in {end_time_ref - start_time_ref:.2f} seconds")

        if reference_variance < VAR_THRESHOLD:
            logger.info(f"Reference variance ({reference_variance:.5f}) is already below the variance threshold ({VAR_THRESHOLD:.5f}). No optimization needed.")
            return {
                "charge_cost_low"    : lowest_charge_cost_low ,
                "charge_cost_high"   : lowest_charge_cost_high,
                "elec_threshold"     : lowest_elec_threshold  ,
            }
        else:
            logger.info(f"Reference variance ({reference_variance:.5f}) is above the variance threshold ({VAR_THRESHOLD:.5f}). Proceeding with optimization.")
            logger.info("Starting Differential Evolution optimization...")

        # Create a partial function with fixed parameters for the worker function
        # Now the worker function only needs the candidate vector as input
        evaluate_single_candidate_worker = partial(
            _evaluate_candidate,
            **kwargs,

            # Pricing Variable bounds
            lower_bounds_a          = lower_bounds_a        ,
            lower_bounds_b          = lower_bounds_b        ,
            lower_bounds_r          = lower_bounds_r        ,
            upper_bounds_r          = upper_bounds_r        ,

            # Metadata
            M                       = M                     ,
            reference_variance      = reference_variance    ,
            log_queue               = log_queue             ,
            log_queue_gurobi        = log_queue_gurobi      ,
        )


        # ----------------------------
        # DE Main Loop
        # ----------------------------
        start_time = time.time()

        # Initial Evaluation
        # We use a Pool to run all candidates in parallel
        logger.info(f"Initializing pool with {NUM_CORES} cores...")

        with multiprocessing.Pool(processes=NUM_CORES) as pool:

            # Calculate initial fitness (Parallel)
            # Runs the evaluate_single_candidate function on each candidate in the population in parallel
            try:
                results: npt.NDArray[np.float64] = np.array(pool.map(evaluate_single_candidate_worker, population))
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

                logger.info(f"Early stopping at initialization with Var = {variances[best_idx_within_qualified]:.5f}, " \
                    f"Fitness = {fitnesses[best_idx_within_qualified]:.5f}, " \
                    f"Var Ratio =  {variance_ratios[best_idx_within_qualified]:.5f}, " \
                    f"Percentage Price Increase = {percentage_price_increases[best_idx_within_qualified]:.5f}%"
                )
                logger.info(f"  Time taken: {init_end_time - start_time:.1f}s")

                return _expand_trajectory(
                    best_vector,
                    M               = M                   ,
                    T               = T                   ,
                    lower_bounds_a  = lower_bounds_a      ,
                    lower_bounds_b  = lower_bounds_b      ,
                    lower_bounds_r  = lower_bounds_r      ,
                    upper_bounds_r  = upper_bounds_r      ,
                    TIMESTEPS       = TIMESTEPS           ,
                )

            # Track both the candidate with best variance (and its fitness), and the candidate with best fitness (and its variance)
            best_var_idx        : int   = np.argmin(variances)
            best_fitness_idx    : int   = np.argmin(fitnesses)

            logger.info(f"Initial candidate with best variance: Var = {variances[best_var_idx]:.5f}, " \
                f"Fitness = {fitnesses[best_var_idx]:.5f}, " \
                f"Var Ratio = {variance_ratios[best_var_idx]:.5f}, " \
                f"Percentage Price Increase = {percentage_price_increases[best_var_idx]:.5f}%"
            )
            logger.info(f"Initial candidate with best fitness: Var = {variances[best_fitness_idx]:.5f}, " \
                f"Fitness = {fitnesses[best_fitness_idx]:.5f}, " \
                f"Var Ratio = {variance_ratios[best_fitness_idx]:.5f}, " \
                f"Percentage Price Increase = {percentage_price_increases[best_fitness_idx]:.5f}%"
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

                    logger.info(f"Early stopping at generation {gen+1} with Var = {variances[best_idx_within_qualified]:.5f}, " \
                        f"Fitness = {fitnesses[best_idx_within_qualified]:.5f}, " \
                        f"Var Ratio = {variance_ratios[best_idx_within_qualified]:.5f}, " \
                        f"Percentage Price Increase = {percentage_price_increases[best_idx_within_qualified]:.5f}%"
                    )
                    logger.info(f"  Time taken: {time.time() - start_time:.1f}s")

                    return _expand_trajectory(
                        candidate_flat  = best_vector       ,
                        M               = M                 ,
                        T               = T                 ,
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
                logger.info(f"  Current candidate with best variance: Var = {variances[best_var_idx]:.5f}, " \
                    f"Fitness = {fitnesses[best_var_idx]:.5f}, " \
                    f"Var Ratio = {variance_ratios[best_var_idx]:.5f}, " \
                    f"Percentage Price Increase = {percentage_price_increases[best_var_idx]:.5f}%"
                )
                logger.info(f"  Current candidate with best fitness: Var = {variances[best_fitness_idx]:.5f}, " \
                    f"Fitness = {fitnesses[best_fitness_idx]:.5f}, " \
                    f"Var Ratio = {variance_ratios[best_fitness_idx]:.5f}, " \
                    f"Percentage Price Increase = {percentage_price_increases[best_fitness_idx]:.5f}%"
                )
                logger.info(f"  Generation time: {gen_duration:.1f}s | Estimated remaining time: {est_remaining_time:.1f}s")


        end_time = time.time()
        logger.info(f"DE completed in {end_time - start_time:.1f}s.")
        logger.info(f"Picking candidate with best fitness...")

        best_vector : npt.NDArray[np.float64] = population[best_fitness_idx].copy()

        return _expand_trajectory(
            candidate_flat  = best_vector       ,
            M               = M                 ,
            T               = T                 ,
            lower_bounds_a  = lower_bounds_a    ,
            lower_bounds_b  = lower_bounds_b    ,
            lower_bounds_r  = lower_bounds_r    ,
            upper_bounds_r  = upper_bounds_r    ,
            TIMESTEPS       = TIMESTEPS         ,
        )

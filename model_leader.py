import os
import numpy as np
import numpy.typing as npt

from logger import Logger
from networkClass import Node, Arc, ArcType

def leader_model(
        **kwargs
    ) -> dict:
    """
    Leader: Charging Operator

    Given EV charging scheduling from the follower, calculate the electrity consumption at each time step,
    the variance of the electricity consumption over the day, and the fitness score.

    Fitness = variance_ratio + penalty_weight * percentage_price_increase
        where
            variance_ratio = variance / reference_variance
            reference_variance = variance when all charging prices are set to minimum levels (a_t = wholesale_elec_price_t, b_t = 0)

            percentage_price_increase = 1/(T-2) sum ((a_t + b_t * K_t - wholesale_elec_price_t)/ wholesale_elec_price_t)
            K_t = 1 - threshold_t / electricity_supplied_t
        
    variance_ratio measures improvement / deterioration in variance of electricity consumption compared to reference variance
    A lower variance_ratio (<1) indicates better load balancing performance
    A higher variance_ratio (>1) indicates worse load balancing performance
    Ideally, it should be minimized to 0 (perfectly flat electricity consumption over time)

    percentage_price_increase penalizes high charging prices set by the leader
    Since setting very high prices (a_t, b_t) may discourage EVs from charging at all, leading to little to no electricity consumption,
    which results in very low variance, giving a misleadingly good fitness score.
    Ideally, it should be as low as possible to 0 while ensuring ideal variance.

    K_t measures the "probability" of overcharging beyond threshold at time t
    A lower threshold_t leads to higher K_t value (between 0 and 1), leading to higher weightage on b_t
    since a_t is base price while b_t is only charged when usage exceeds threshold (on top of a_t)
    If no electricity is supplied at time t (electricity_supplied_t = 0), then K_t = 0, so no penalty on b_t in this case
    
    Returns a dictionary containing:
        - fitness: float
        - variance: float
        - variance_ratio: float
        - percentage_price_increase: float
    """
    # ----------------------------
    # Parameters
    # ----------------------------
    
    # Follower parameters
    T                       : int                           = kwargs.get("T")                       # termination time of daily operations (0, ..., T)
    elec_supplied           : dict[tuple[int, int], int]    = kwargs.get("elec_supplied")           # electricity supplied (in SoC levels) at zone i at time t
        
    charge_cost_low         : dict[int, float]              = kwargs.get("charge_cost_low")         # charge cost per unit of SoC at zone i at time t when usage is below threshold
    charge_cost_high        : dict[int, float]              = kwargs.get("charge_cost_high")        # charge cost per unit of SOC at zone i at time t when usage is above threshold
    elec_threshold          : dict[int, int]                = kwargs.get("elec_threshold")          # electricity threshold at zone i at time t

    # leader parameters
    penalty_weight          : float                         = kwargs.get("penalty_weight")          # weight for penalty on high price settings
    wholesale_elec_price    : dict[int, float]              = kwargs.get("wholesale_elec_price")    # wholesale electricity price at time t ($ per kWh)
    reference_variance      : float                         = kwargs.get("reference_variance")      # reference variance for normalization

    # Output solutions
    x                       : dict[int, float]              = kwargs.get("x", {})

    # Arcs
    all_arcs                : dict[int, Arc]                = kwargs["all_arcs"]
    charge_arcs_t           : dict[int, set[int]]           = kwargs["charge_arcs_t"]

    # Sets
    ZONES                   : list[int]                     = kwargs["ZONES"]

    # Metadata
    to_console              : bool                          = kwargs.get("to_console", False)       # whether to log to console
    to_file                 : bool                          = kwargs.get("to_file", True)           # whether to log to file
    timestamp               : str                           = kwargs.get("timestamp", "")           # timestamp for logging
    file_name               : str                           = kwargs.get("file_name", "")           # filename for logging
    folder_name             : str                           = kwargs.get("folder_name", "")         # folder name for logs and results

    # Create logger
    logger = Logger("model_leader", level="DEBUG", to_console=to_console, timestamp=timestamp)
    if to_file:
        logger.save (os.path.join (folder_name, f"model_leader_{file_name}"))

    logger.info("Parameters loaded successfully into leader model.")


    # ----------------------------
    # Variance Calculation
    # ----------------------------

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
    usage_vector        : npt.NDArray[np.float_] = electricity_usage[1:T]  # exclude time 0 and T
    mean_usage          : float                  = np.mean(usage_vector)
    variance            : float                  = np.var(usage_vector, ddof=1) if len(usage_vector) > 1 else 0.0
    variance_ratio      : float                  = variance / reference_variance if reference_variance > 0 else 0.0


    # ----------------------------
    # Fitness Calculation
    # ----------------------------
    
    # Convert dictionaries to numpy arrays for vectorized operations
    timesteps_range             : npt.NDArray[np.int_]   = np.arange(1, T)  # exclude time 0 and T
    
    # Pre-compute electricity supplied at each time step
    electricity_supplied_arr    : npt.NDArray[np.float_] = np.array([
        sum(elec_supplied.get((i, t), 0) for i in ZONES) for t in timesteps_range
    ])
    
    # Convert pricing arrays
    charge_cost_low_arr         : npt.NDArray[np.float_] = np.array([charge_cost_low[t] for t in timesteps_range])
    charge_cost_high_arr        : npt.NDArray[np.float_] = np.array([charge_cost_high[t] for t in timesteps_range])
    elec_threshold_arr          : npt.NDArray[np.float_] = np.array([elec_threshold[t] for t in timesteps_range])
    wholesale_elec_price_arr    : npt.NDArray[np.float_] = np.array([wholesale_elec_price[t] for t in timesteps_range])
    
    # Calculate K values vectorized (avoid division by zero)
    K_values                    : npt.NDArray[np.float_] = np.where(
        electricity_supplied_arr > 0,
        1 - elec_threshold_arr / electricity_supplied_arr,
        0.0
    )
    
    # Calculate price increases vectorized
    price_increases             : npt.NDArray[np.float_] = (
        (charge_cost_low_arr + charge_cost_high_arr * K_values) - wholesale_elec_price_arr
    ) / wholesale_elec_price_arr
    
    percentage_price_increase   : float = np.mean(price_increases)
    fitness                     : float = variance_ratio + penalty_weight * percentage_price_increase

    logger.info("Leader model computation completed.")
    return fitness, variance, variance_ratio, percentage_price_increase

        
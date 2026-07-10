import numpy as np
import numpy.typing as npt

from logger import Logger
from networkClass import Arc
from config_DE import (
    PENALTY_WEIGHT  ,
    STRIDE          ,
    WINDOW_SIZE     ,
)

def leader_model(
        **kwargs
    ) -> dict:
    """
    Leader: Charging Operator

    Given EV charging scheduling from the follower, calculate the electrity consumption at each time step,
    the local variances of electricity consumption over a sliding window, the local variance ratios compared to reference variances, and the overall fitness score.

    Fitness = variance_ratio + penalty_weight * percentage_price_increase <br>
    Where: \n
        - variance_ratio = sum (local_variance / local_reference_variance)
        - reference_variance = variance when all charging prices are set to minimum levels (a_t = wholesale_elec_price_t, b_t = 0)

        - percentage_price_increase = 1/(T-2) sum ((a_t + b_t * K_t - wholesale_elec_price_t)/ wholesale_elec_price_t)
        - K_t = 1 - threshold_t / electricity_supplied_t
        
    variance_ratio measures improvement / deterioration in variance of electricity consumption compared to reference variance <br>
    A lower variance_ratio (<1) indicates better load balancing performance <br>
    A higher variance_ratio (>1) indicates worse load balancing performance <br>
    Ideally, it should be minimized to 0 (perfectly flat electricity consumption over time)

    percentage_price_increase penalizes high charging prices set by the leader <br>
    Since setting very high prices (a_t, b_t) may discourage EVs from charging at all, leading to little to no electricity consumption, <br>
    which results in very low variance, giving a misleadingly good fitness score. <br>
    Ideally, it should be as low as possible to 0 while ensuring ideal variance. 

    K_t measures the "probability" of overcharging beyond threshold at time t <br>
    A lower threshold_t leads to higher K_t value (between 0 and 1), leading to higher weightage on b_t <br>
    since a_t is base price while b_t is only charged when usage exceeds threshold (on top of a_t) <br>
    If no electricity is supplied at time t (electricity_supplied_t = 0), then K_t = 0, so no penalty on b_t in this case
    
    Returns a dictionary containing: \n
        - fitness: float
        - variance_ratio: float
        - percentage_price_increase: float
    """
    # ----------------------------
    # Parameters
    # ----------------------------
    # Follower model parameters
    T                       : int                               = kwargs["T"]                       # termination time of daily operations (0, ..., T)
    elec_supplied           : dict[tuple[int, int]  , int]      = kwargs["elec_supplied"]           # electricity supplied (in SoC levels) at zone i at time t

    # Network components
    all_arcs                : dict[int              , Arc]      = kwargs["all_arcs"]
    charge_arcs_t           : dict[int              , set[int]] = kwargs["charge_arcs_t"]
    ZONES                   : list[int]                         = kwargs["ZONES"]
    TIMESTEPS               : list[int]                         = kwargs["TIMESTEPS"]

    # Leader model parameters
    wholesale_elec_price    : dict[int              , float]    = kwargs["wholesale_elec_price"]    # wholesale electricity price at time t
    reference_variances     : dict[int, float]                  = kwargs["reference_variances"]     # reference variance for normalization
    reference_usage         : float                             = kwargs["reference_usage"]         # reference electricity usage for normalization

    # Pricing Variables
    charge_cost_low         : dict[int              , float]    = kwargs["charge_cost_low"]         # a_t
    charge_cost_high        : dict[int              , float]    = kwargs["charge_cost_high"]        # b_t
    elec_threshold          : dict[int              , int]      = kwargs["elec_threshold"]          # r_t

    # Solutions
    x                       : dict[int              , float]    = kwargs["x"]

    # Metadata
    logger                  : Logger                            = kwargs["logger"]                  # logger instance
    was_suboptimal          : bool                              = kwargs["was_suboptimal"]          # whether the follower model was suboptimal

    # ---------------------------------------
    # Percentage Usage Decrease Calculation
    # ---------------------------------------
    # Calculate electricity consumption at each time step using vectorized operations
    # Exclude first time step (t=0) as no charging occurs at t=0
    # Exclude last time step (t=T) as no charging occurs at t=T
    electricity_usage: npt.NDArray[np.float64] = np.zeros(T - 1) # electricity usage from t=1 to t=T-1

    for t in TIMESTEPS[1:-1]: 
        # Calculate total electricity used at time t
        for e_id in charge_arcs_t.get(t, set()):
            arc = all_arcs[e_id]

            # Electricity used = number of EVs * charge amount
            charge_amount = arc.d.l - arc.o.l  # SoC levels charged
            electricity_usage[t - 1] += x[e_id] * charge_amount
    
    current_usage               : float = np.sum(electricity_usage)
    percentage_usage_decrease   : float = (reference_usage - current_usage) / (reference_usage + 1e-8)  # Add a small epsilon to avoid division by zero

    logger.info(f"Leader model electricity usage calculation: Current Usage = {current_usage:.3f}, Reference Usage = {reference_usage:.3f}, Percentage Usage Decrease = {percentage_usage_decrease:.1%}")


    # ---------------------------------------
    # Variance Ratio Calculation
    # ---------------------------------------
    # Calculate local variances for each window of electricity usage
    local_variances: dict[int, float] = {}
    for start in range(0, len(electricity_usage), STRIDE):
        end = min(start + WINDOW_SIZE, len(electricity_usage))
        window = electricity_usage[start:end]
        if len(window) >= 2:
            local_variances[start] = np.var(window, ddof=0)  # ddof=0 for population variance

        # if the end of the window is at the end of the usage_vector, we break the loop since we cannot form any more windows
        if end == len(electricity_usage):
            break

    # Check if key matches between local_variances and reference_variances
    if set(local_variances.keys()) != set(reference_variances.keys()):
        logger.error(f"Mismatch in keys between local_variances and reference_variances. Local keys: {list(local_variances.keys())}, Reference keys: {list(reference_variances.keys())}")
        raise KeyError("Mismatch in keys between local_variances and reference_variances.")

    # Calculate the variance ratio at each time step using the reference variances, then sum them
    variance_ratios: list[float] = []
    for start, local_variance in local_variances.items():
        
        reference_variance  : float = reference_variances[start]
        variance_ratio      : float = local_variance / (reference_variance + 1e-8)  # Add a small epsilon to avoid division by zero
        variance_ratios.append(variance_ratio)

    variance_ratio: float = np.sum(variance_ratios) 

    logger.info(f"Leader model variance ratio calculation: Variance Ratio = {variance_ratio:.3f}")


    # ---------------------------------------
    # Percentage Price Increase Calculation
    # ---------------------------------------
    # Pre-compute electricity supplied at each time step (excluding first and last)
    electricity_supplied_arr    : npt.NDArray[np.float64] = np.array([
        sum(elec_supplied.get((i, t), 0) for i in ZONES) for t in TIMESTEPS[1:-1]
    ])
    
    # Convert pricing arrays
    charge_cost_low_arr         : npt.NDArray[np.float64] = np.array([charge_cost_low[t]        for t in TIMESTEPS[1:-1]])
    charge_cost_high_arr        : npt.NDArray[np.float64] = np.array([charge_cost_high[t]       for t in TIMESTEPS[1:-1]])
    elec_threshold_arr          : npt.NDArray[np.float64] = np.array([elec_threshold[t]         for t in TIMESTEPS[1:-1]])
    wholesale_elec_price_arr    : npt.NDArray[np.float64] = np.array([wholesale_elec_price[t]   for t in TIMESTEPS[1:-1]])
    
    # Calculate K values vectorized (avoid division by zero)
    K_values                    : npt.NDArray[np.float64] = np.where(
        electricity_supplied_arr > 0,
        1 - elec_threshold_arr / electricity_supplied_arr,
        0.0
    )
    
    # Calculate price increases vectorized
    price_increases             : npt.NDArray[np.float64] = (
        (charge_cost_low_arr + charge_cost_high_arr * K_values) - wholesale_elec_price_arr
    ) / wholesale_elec_price_arr
    
    percentage_price_increase   : float = np.mean(price_increases)
    logger.info(f"Leader model percentage price increase calculation: Percentage Price Increase = {percentage_price_increase:.1%}")


    # ---------------------------------------
    # Fitness Calculation
    # ---------------------------------------
    fitness                     : float = variance_ratio + PENALTY_WEIGHT * percentage_price_increase
    logger.info(f"Leader model completed. Fitness: {fitness:.3f}")


    return {
        "fitness"                   : fitness                   ,
        "variance_ratio"            : variance_ratio            ,
        "percentage_price_increase" : percentage_price_increase ,
        "percentage_usage_decrease" : percentage_usage_decrease ,
        "was_suboptimal"            : was_suboptimal
    }

        
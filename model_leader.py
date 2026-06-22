import numpy as np
import numpy.typing as npt

from logger import Logger
from networkClass import Arc
from config_DE import PENALTY_WEIGHT

def leader_model(
        **kwargs
    ) -> dict:
    """
    Leader: Charging Operator

    Given EV charging scheduling from the follower, calculate the electrity consumption at each time step,
    the ramp rate of the electricity consumption over the day, and the fitness score.

    Fitness = ramp_rate_ratio + penalty_weight * percentage_price_increase <br>
    Where: \n
        - ramp_rate_ratio = ramp_rate / reference_ramp_rate
        - reference_ramp_rate = ramp rate when all charging prices are set to minimum levels (a_t = wholesale_elec_price_t, b_t = 0)

        - percentage_price_increase = 1/(T-2) sum ((a_t + b_t * K_t - wholesale_elec_price_t)/ wholesale_elec_price_t)
        - K_t = 1 - threshold_t / electricity_supplied_t
        
    ramp_rate_ratio measures improvement / deterioration in ramp rate of electricity consumption compared to reference ramp rate <br>
    A lower ramp_rate_ratio (<1) indicates better load balancing performance <br>
    A higher ramp_rate_ratio (>1) indicates worse load balancing performance <br>
    Ideally, it should be minimized to 0 (perfectly flat electricity consumption over time)

    percentage_price_increase penalizes high charging prices set by the leader <br>
    Since setting very high prices (a_t, b_t) may discourage EVs from charging at all, leading to little to no electricity consumption, <br>
    which results in very low ramp rate, giving a misleadingly good fitness score. <br>
    Ideally, it should be as low as possible to 0 while ensuring ideal ramp rate. 

    K_t measures the "probability" of overcharging beyond threshold at time t <br>
    A lower threshold_t leads to higher K_t value (between 0 and 1), leading to higher weightage on b_t <br>
    since a_t is base price while b_t is only charged when usage exceeds threshold (on top of a_t) <br>
    If no electricity is supplied at time t (electricity_supplied_t = 0), then K_t = 0, so no penalty on b_t in this case
    
    Returns a dictionary containing: \n
        - fitness: float
        - ramp_rate: float
        - ramp_rate_ratio: float
        - percentage_price_increase: float
    """
    # ----------------------------
    # Parameters
    # ----------------------------
    # Follower model parameters
    T                       : int                                       = kwargs["T"]                       # termination time of daily operations (0, ..., T)
    elec_supplied           : dict[tuple[int, int]          , int]      = kwargs["elec_supplied"]           # electricity supplied (in SoC levels) at zone i at time t

    # Network components
    all_arcs                : dict[int                      , Arc]      = kwargs["all_arcs"]
    charge_arcs_t           : dict[int                      , set[int]] = kwargs["charge_arcs_t"]
    ZONES                   : list[int]                                 = kwargs["ZONES"]
    TIMESTEPS               : list[int]                                 = kwargs["TIMESTEPS"]

    # Leader model parameters
    wholesale_elec_price    : dict[int                      , float]    = kwargs["wholesale_elec_price"]    # wholesale electricity price at time t
    reference_ramp_rate     : float                                     = kwargs["reference_ramp_rate"]     # reference ramp rate for normalization
    reference_total_usage   : float                                     = kwargs["total_usage"]             # total electricity usage across all time steps

    # Pricing Variables
    charge_cost_low         : dict[int                      , float]    = kwargs["charge_cost_low"]         # a_t
    charge_cost_high        : dict[int                      , float]    = kwargs["charge_cost_high"]        # b_t
    elec_threshold          : dict[int                      , int]      = kwargs["elec_threshold"]          # r_t

    # Solutions
    x                       : dict[int                      , float]    = kwargs["x"]

    # Metadata
    logger                  : Logger                                    = kwargs["logger"]                  # logger instance
    was_suboptimal          : bool                                      = kwargs["was_suboptimal"]          # whether the follower model was suboptimal

    # ----------------------------
    # Ramp Rate Calculation
    # ----------------------------

    # Calculate electricity consumption at each time step using vectorized operations
    # Exclude first time step (t=0) as no charging occurs at t=0
    # Exclude last time step (t=T) as no charging occurs at t=T
    electricity_usage   : npt.NDArray[np.float64] = np.zeros(T - 1) # electricity usage from t=1 to t=T-1
    current_total_usage : float = electricity_usage.sum()  # total electricity usage across all time steps (excluding t=0 and t=T)

    for t in TIMESTEPS[1:-1]: 
        # Calculate total electricity used at time t
        for e_id in charge_arcs_t.get(t, set()):
            arc = all_arcs[e_id]

            # Electricity used = number of EVs * charge amount
            charge_amount = arc.d.l - arc.o.l  # SoC levels charged
            electricity_usage[t - 1] += x[e_id] * charge_amount

    # Calculate ramp rate of electricity consumption using numpy
    ramp_rate                   : float                     = np.sum(np.diff(electricity_usage)**2) if len(electricity_usage) > 1 else 0.0
    ramp_rate_ratio             : float                     = ramp_rate / (reference_ramp_rate + 1e-6)  # Add small epsilon to avoid division by zero

    percentage_usage_decrease   : float                     = (reference_total_usage - current_total_usage) / (reference_total_usage + 1e-6)  # Add small epsilon to avoid division by zero

    logger.info(f"Leader model ramp rate calculation: Ramp Rate = {ramp_rate:.3f}, Ramp Rate Ratio = {ramp_rate_ratio:.3f}, % Usage Decrease = {percentage_usage_decrease:.3f}%")

    # ----------------------------
    # Fitness Calculation
    # ----------------------------    
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
    # fitness                     : float = ramp_rate_ratio + PENALTY_WEIGHT * percentage_price_increase
    fitness                     : float = ramp_rate_ratio + PENALTY_WEIGHT * percentage_usage_decrease

    logger.info(f"Leader model completed. Fitness: {fitness:.3f}, % Price Increase: {percentage_price_increase:.3f}%")

    return {
        "fitness"                   : fitness                   ,
        "ramp_rate"                 : ramp_rate                 ,
        "ramp_rate_ratio"           : ramp_rate_ratio           ,
        "percentage_price_increase" : percentage_price_increase ,
        "percentage_usage_decrease" : percentage_usage_decrease ,
        "was_suboptimal"            : was_suboptimal
    }

        
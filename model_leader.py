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
    # Follower model parameters
    N                       : int                                       = kwargs["N"]                       # number of operation zones (1, ..., N)
    T                       : int                                       = kwargs["T"]                       # termination time of daily operations (0, ..., T)
    L                       : int                                       = kwargs["L"]                       # max SoC level (all EVs start at this level) (0, ..., L)
    W                       : int                                       = kwargs["W"]                       # maximum time intervals a passenger will wait for a ride (0, ..., W-1; demand expires at W)
    travel_demand           : dict[tuple[int, int, int]     , int]      = kwargs["travel_demand"]           # travel demand from zone i to j at starting at time t
    travel_time             : dict[tuple[int, int, int]     , int]      = kwargs["travel_time"]             # travel time from i to j at starting at time t
    travel_energy           : dict[tuple[int, int]          , int]      = kwargs["travel_energy"]           # energy consumed for trip from zone i to j
    order_revenue           : dict[tuple[int, int, int]     , float]    = kwargs["order_revenue"]           # order revenue for each trip served from i to j at time t
    penalty                 : dict[tuple[int, int, int]     , float]    = kwargs["penalty"]                 # penalty cost for each unserved trip from i to j at time t
    L_min                   : int                                       = kwargs["L_min"]                   # min SoC level all EV must end with at the end of the daily operations
    num_EVs                 : int                                       = kwargs["num_EVs"]                 # total number of EVs in the fleet 
    num_ports               : dict[int                      , int]      = kwargs["num_ports"]               # number of charging ports in each zone
    elec_supplied           : dict[tuple[int, int]          , int]      = kwargs["elec_supplied"]           # electricity supplied (in SoC levels) at zone i at time t
    max_charge_speed        : int                                       = kwargs["max_charge_speed"]        # max charging speed (in SoC levels) of one EV in one time step

    # Network components
    V_set                   : set[Node]                                 = kwargs["V_set"]
    all_arcs                : dict[int                      , Arc]      = kwargs["all_arcs"]
    type_arcs               : dict[ArcType                  , set[int]] = kwargs["type_arcs"]
    in_arcs                 : dict[Node                     , set[int]] = kwargs["in_arcs"]
    out_arcs                : dict[Node                     , set[int]] = kwargs["out_arcs"]
    service_arcs_ijt        : dict[tuple[int, int, int]     , set[int]] = kwargs["service_arcs_ijt"]
    charge_arcs_it          : dict[tuple[int, int]          , set[int]] = kwargs["charge_arcs_it"]
    charge_arcs_t           : dict[int                      , set[int]] = kwargs["charge_arcs_t"]
    valid_travel_demand     : dict[tuple[int, int, int]     , int]      = kwargs["valid_travel_demand"]
    invalid_travel_demand   : set[tuple[int, int, int]]                 = kwargs["invalid_travel_demand"]
    ZONES                   : list[int]                                 = kwargs["ZONES"]
    TIMESTEPS               : list[int]                                 = kwargs["TIMESTEPS"]
    LEVELS                  : list[int]                                 = kwargs["LEVELS"]
    AGES                    : list[int]                                 = kwargs["AGES"]

    # Leader model parameters
    wholesale_elec_price    : dict[int                      , float]    = kwargs["wholesale_elec_price"]    # wholesale electricity price at time t
    PENALTY_WEIGHT          : float                                     = kwargs["PENALTY_WEIGHT"]          # penalty weight for high a_t and b_t
    reference_variance      : float                                     = kwargs["reference_variance"]      # reference variance for normalization

    # Pricing Variables
    charge_cost_low         : dict[int                      , float]    = kwargs["charge_cost_low"]         # a_t
    charge_cost_high        : dict[int                      , float]    = kwargs["charge_cost_high"]        # b_t
    elec_threshold          : dict[int                      , int]      = kwargs["elec_threshold"]          # r_t

    # Solutions
    obj                     : float                                     = kwargs["obj"]
    x                       : dict[int                      , float]    = kwargs["x"]
    s                       : dict[tuple[int, int, int]     , float]    = kwargs["s"]
    u                       : dict[tuple[int, int, int, int], float]    = kwargs["u"]
    e                       : dict[tuple[int, int, int]     , float]    = kwargs["e"]
    q                       : dict[int                      , float]    = kwargs["q"]
    service_revenues        : dict[int                      , float]    = kwargs["service_revenues"]
    penalty_costs           : dict[int                      , float]    = kwargs["penalty_costs"]
    charge_costs            : dict[int                      , float]    = kwargs["charge_costs"]

    # Metadata
    timestamp               : str                                       = kwargs["timestamp"]                   # timestamp for logging
    file_name               : str                                       = kwargs["file_name"]                   # filename for logging
    folder_name             : str                                       = kwargs["folder_name"]                 # folder name for logs and results
    logger                  : Logger                                    = kwargs["logger"]                      # logger instance

    # ----------------------------
    # Variance Calculation
    # ----------------------------

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
    usage_vector        : npt.NDArray[np.float64]   = electricity_usage[1:T]  # exclude time 0 and T
    variance            : float                     = np.var(usage_vector, ddof=0) if len(usage_vector) > 1 else 0.0   # ddof=0 for population variance, =1 for sample variance
    variance_ratio      : float                     = variance / reference_variance if reference_variance > 0 else 0.0

    logger.info(f"8/10: Leader model variance calculation: Variance = {variance:.4f}, Variance Ratio = {variance_ratio:.4f}")

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
    fitness                     : float = variance_ratio + PENALTY_WEIGHT * percentage_price_increase

    logger.info(f"9/10: Leader model completed. Fitness: {fitness:.4f}, Percentage Price Increase: {percentage_price_increase:.4f}")

    return {
        "fitness"                   : fitness,
        "variance"                  : variance,
        "variance_ratio"            : variance_ratio,
        "percentage_price_increase" : percentage_price_increase
    }

        
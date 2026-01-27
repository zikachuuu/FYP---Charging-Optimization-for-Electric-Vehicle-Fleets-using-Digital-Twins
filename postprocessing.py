import pandas as pd
import matplotlib.pyplot as plt
import os

from logger import Logger
from networkClass import Node, Arc, ArcType

def postprocessing(**kwargs):
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

    # Calculated information
    service_revenues        : dict[int                      , float]    = kwargs["service_revenues"]
    penalty_costs           : dict[int                      , float]    = kwargs["penalty_costs"]
    charge_costs            : dict[int                      , float]    = kwargs["charge_costs"]
    total_service_revenue   : float                                     = kwargs["total_service_revenue"]
    total_penalty_cost      : float                                     = kwargs["total_penalty_cost"]
    total_charge_cost       : float                                     = kwargs["total_charge_cost"]

    # Metadata
    timestamp               : str                                       = kwargs["timestamp"]               # timestamp for logging
    file_name               : str                                       = kwargs["file_name"]               # filename for logging
    folder_name             : str                                       = kwargs["folder_name"]             # folder name for logs and results
    results_name            : str                                       = kwargs["results_name"]

    # markers for plot
    markers = ["o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_"]

    logger = Logger(
        "postprocessing"            ,
        level       = "DEBUG"       ,
        to_console  = True          ,
        folder_name = folder_name   ,
        file_name   = file_name     ,
        timestamp   = timestamp     ,
    )
    logger.info("Parameters loaded successfully")
    
    
    def _fleet_operator_profit():
        """
        Build a DataFrame with profit breakdown by time:
        - Rows: time intervals (0 to T) plus a 'total' row
        - Columns: Profit, Service Revenue, Penalty Cost, Charging Cost
        - Values: metrics for each time step (not cumulative)
        
        Also creates a plot showing all metrics over time.
        
        Returns:
            pd.DataFrame: Profit breakdown by time
        """
        df_profit = pd.DataFrame({
            "Profit"            : [service_revenues[t] - penalty_costs[t] - charge_costs[t] for t in TIMESTEPS] ,
            "Service Revenue"   : service_revenues.values()                                                     ,
            "Penalty Cost"      : penalty_costs.values()                                                        ,
            "Charging Cost"     : charge_costs.values()
        }, index=TIMESTEPS)

        total_row = pd.DataFrame({
            "Profit"            : obj                       ,
            "Service Revenue"   : [total_service_revenue]   ,
            "Penalty Cost"      : [total_penalty_cost]      ,
            "Charging Cost"     : [total_charge_cost]
        }, index=["total"])

        df_profit = pd.concat([df_profit, total_row])
        df_profit.index.name = "Time Interval"

        # Create plot (excluding total row)
        df_plot = df_profit.iloc[:-1] # exclude total row for plotting
        plt.figure(figsize=(10, 5))

        # Plot each metric
        plt.plot(
            df_plot.index, 
            df_plot["Profit"]               , 
            color       = 'tab:green'       , 
            linewidth   = 2.5               , 
            marker      = markers[0]        , 
            label       = 'Profit'          ,
        )
        plt.plot(
            df_plot.index                   , 
            df_plot["Service Revenue"]      , 
            color       = 'tab:blue'        , 
            linewidth   = 2                 , 
            marker      = markers[1]        ,
            label       = 'Service Revenue' ,
        )
        plt.plot(
            df_plot.index                   , 
            df_plot["Penalty Cost"]         , 
            color       = 'tab:red'         , 
            linewidth   = 2                 , 
            marker      = markers[2]        ,
            label       = 'Penalty Cost'    ,
        )
        plt.plot(
            df_plot.index                   , 
            df_plot["Charging Cost"]        , 
            color       = 'tab:orange'      , 
            linewidth   = 2                 , 
            marker      = markers[3]        ,
            label       = 'Charging Cost'   ,
        )
        
        plt.xlabel  ('Time Intervals')
        plt.ylabel  ('Money ($)')
        plt.title   ('Fleet Operator Profit Breakdown Over Time')
        plt.grid    (True, alpha=0.3)
        plt.legend  (loc='best', frameon=False)

        plt.tight_layout()
        
        outfile = os.path.join ("Results", folder_name, f"fleet_operator_profit_{file_name}_{timestamp}.png")
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved Fleet Operator Profit plot to {outfile}")
        
        return df_profit

    def _calculate_ev_operations_by_type():
        """
        Calculate EV operations counts by arc type over time (excluding WRAP).
        
        Returns:
            dict: Operations count by arc type and time
        """
        ev_operations = {arc_type: [0] * (T+1) for arc_type in ArcType if arc_type != ArcType.WRAP}
        for type, e_ids in type_arcs.items():
            if type == ArcType.WRAP:
                continue
            for e_id in e_ids:
                arc = all_arcs[e_id]
                start_t = arc.o.t
                end_t = arc.d.t
                for t in range(start_t, end_t):
                    ev_operations[type][t] += x[e_id]
        return ev_operations

    def _ev_operations_over_time():
        """
        Build a DataFrame with EV operations percentage by type over time.
        Also creates and saves a plot.
        
        Returns:
            pd.DataFrame: EV operations percentage by type over time
        """
        ev_operations = _calculate_ev_operations_by_type()
        
        df_ev_operations = pd.DataFrame(ev_operations, index=TIMESTEPS)
        df_ev_operations = df_ev_operations.div(num_EVs).mul(100)  # convert to percentage
        df_ev_operations.index.name = "Time Interval"
        df_ev_operations.columns = [
            f"Percentage of {col.name} EVs" 
            if hasattr(col, "name") 
            else f"Percentage of {str(col)} EVs" 
            for col in df_ev_operations.columns
        ]

        # Plot
        plt.figure(figsize=(10, 5))
        type_colors = {
            "SERVICE"   : 'tab:blue'   ,
            "CHARGE"    : 'tab:orange' ,
            "IDLE"      : 'tab:green'  ,
            "RELOCATION": 'tab:red'    ,
        }

        for col in df_ev_operations.columns:
            label = col.split("Percentage of ")[1].split(" EVs")[0]  # Extract arc type name
            marker = markers[df_ev_operations.columns.get_loc(col) % len(markers)]
            plt.plot(
                df_ev_operations.index  , 
                df_ev_operations[col]   , 
                marker      = marker    , 
                linewidth   = 2         , 
                label       = label     ,
                color       = type_colors.get(label, None),
            )

        plt.xlabel  ("Time Intervals")
        plt.ylabel  ("% of EVs")
        plt.title   ("SAEV Operations by Activities over Time")
        plt.grid    (True, alpha=0.3)
        plt.legend  (title="Activity", loc="best", frameon=False)
        plt.tight_layout()

        outfile = os.path.join ("Results", folder_name, f"ev_operations_{file_name}_{timestamp}.png")
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved EV Operations plot to {outfile}")

        return df_ev_operations

    def _calculate_demand_metrics():
        """
        Calculate demand metrics for all time steps.
        
        Returns:
            list: Demand metrics by time step
        """
        demand_served = [{} for _ in range(T + 2)]  # Demand served at each time t, from t=0 to t=T, last row for total
        
        for t in TIMESTEPS:
            # Calculate metrics for each OD pair
            for i in ZONES:
                for j in ZONES:
                    if i == j:
                        continue

                    new_demand      = valid_travel_demand.get((i, j, t), 0)
                    served_demand   = sum(x[e_id] for e_id in service_arcs_ijt.get((i, j, t), set()))
                    unserved_demand = s[(i, j, t)]
                    expired_demand  = e[(i, j, t)]

                    demand_served[t][(i, j)] = {
                        "new demand"                : new_demand,
                        "served demand"             : served_demand,
                        "remaining unserved demand" : unserved_demand,
                        "expired demand"            : expired_demand
                    }

                    for age in AGES:
                        demand_served[t][(i, j)][f"unserved demand of age {age}"] = u[(i, j, t, age)]
            
            # Calculate total metrics for this time step
            total_new_demand_by_time = sum(
                valid_travel_demand.get((i, j, t), 0)
                for i in ZONES 
                for j in ZONES
            )
            total_served_by_time = sum( 
                x[e] 
                for i in ZONES 
                for j in ZONES 
                for e in service_arcs_ijt.get((i, j, t), set()) 
            )
            total_unserved_by_time = sum(
                s[(i, j, t)] 
                for i in ZONES 
                for j in ZONES
            )
            total_expired_by_time = sum(
                e[(i, j, t)]
                for i in ZONES
                for j in ZONES
            )
            
            demand_served[t]["total"] = {
                "new demand"                : total_new_demand_by_time,
                "served demand"             : total_served_by_time,
                "remaining unserved demand" : total_unserved_by_time,
                "expired demand"            : total_expired_by_time
            }
            for age in AGES:
                demand_served[t]["total"][f"unserved demand of age {age}"] = sum(
                    u[(i, j, t, age)]
                    for i in ZONES
                    for j in ZONES
                )
        
        # Last row: total over all time intervals
        for i in ZONES:
            for j in ZONES:
                if i == j:
                    continue
                total_new_demand_by_zone = sum(
                    valid_travel_demand.get((i, j, t), 0)
                    for t in TIMESTEPS
                )
                total_served_by_zone = sum(
                    x[e]
                    for t in TIMESTEPS
                    for e in service_arcs_ijt.get((i, j, t), set())
                )
                total_unserved_by_zone = 0     # all unserved are expired
                total_expired_by_zone = sum(
                    e[(i, j, t)]
                    for t in TIMESTEPS
                ) + s[(i, j, T)]

                demand_served[T + 1][(i, j)] = {
                    "new demand":                   total_new_demand_by_zone,
                    "served demand":                total_served_by_zone,
                    "remaining unserved demand":    total_unserved_by_zone,
                    "expired demand":               total_expired_by_zone
                }
                for age in AGES:
                    demand_served[T + 1][(i, j)][f"unserved demand of age {age}"] = 0
        
        # Calculate overall totals
        total_new_demand = sum(valid_travel_demand.values())
        total_served = sum(x.get(arc_id, 0) for arc_id in type_arcs[ArcType.SERVICE])
        total_unserved = 0
        total_expired = sum(
            e[(i, j, t)]
            for i in ZONES
            for j in ZONES
            for t in TIMESTEPS
        ) + sum(
            s[(i, j, T)]
            for i in ZONES
            for j in ZONES
        )
        
        demand_served[T + 1]["total"] = {
            "new demand":                   total_new_demand,
            "served demand":                total_served,
            "remaining unserved demand":    total_unserved,
            "expired demand":               total_expired
        }
        for age in AGES:
            demand_served[T + 1]["total"][f"unserved demand of age {age}"] = 0
        
        return demand_served

    def _demand_served_to_dataframe(demand_served):
        """
        Convert demand_served list/dict structure to a DataFrame.
        
        Args:
            demand_served: List of demand metrics by time step
            
        Returns:
            pd.DataFrame: Formatted demand served data
        """
        # Determine all (i, j) pairs and metrics
        ij_pairs = [(i, j) for i in ZONES for j in ZONES if i != j]
        base_metrics = [
            "new demand",
            "served demand",
            "remaining unserved demand",
            "expired demand",
        ]
        age_metrics = [f"unserved demand of age {age}" for age in AGES]
        metrics = base_metrics + age_metrics

        # Build column MultiIndex
        main_cols = ij_pairs + ["total"]
        cols = pd.MultiIndex.from_product([main_cols, metrics], names=["(Start Zone, End Zone)", "metric"])

        # Index: all t plus final 'total'
        index = list(TIMESTEPS) + ["total"]

        # Initialize DataFrame
        df = pd.DataFrame(index=index, columns=cols, dtype=float)

        # Fill values from demand_served
        for t in TIMESTEPS:
            row_key = t
            row_dict = demand_served[t]
            for od in main_cols:
                od_dict = row_dict.get(od, {})
                for m in metrics:
                    df.loc[row_key, (od, m)] = od_dict.get(m, 0)

        # Final total row
        total_row = demand_served[T + 1]
        for od in main_cols:
            od_dict = total_row.get(od, {})
            for m in metrics:
                df.loc["total", (od, m)] = od_dict.get(m, 0)

        return df

    def _demand_served_over_time():
        """
        Build a DataFrame with demand metrics over time.
        
        Returns:
            pd.DataFrame: Demand served metrics over time
        """
        demand_served = _calculate_demand_metrics()
        df_demand_served = _demand_served_to_dataframe(demand_served)
        return df_demand_served

    def _service_and_demand_comparison():
        """
        Create a plot comparing SAEV service operations with demand metrics.
        Left y-axis: percentage of SAEVs in service
        Right y-axis: number of rides (new, cumulative unfulfilled, expired)
        
        Returns:
            pd.DataFrame: Combined metrics for service and demand
        """
        # Get EV operations data
        ev_operations = _calculate_ev_operations_by_type()
        service_percentage = [op / num_EVs * 100 for op in ev_operations[ArcType.SERVICE]]
        
        # Get demand metrics
        demand_metrics = _calculate_demand_metrics()
        
        # Extract total metrics over time
        new_demand                  = []
        fulfilled                   = []
        unfulfilled                 = []
        cumulative_expired_demand   = []
        
        for t in TIMESTEPS:
            new_demand.append               (demand_metrics[t]["total"]["new demand"])
            fulfilled.append                (demand_metrics[t]["total"]["served demand"])
            unfulfilled.append              (demand_metrics[t]["total"]["remaining unserved demand"])
            cumulative_expired_demand.append(demand_metrics[t]["total"]["expired demand"] + (cumulative_expired_demand[t-1] if t > 0 else 0))
        
        # Create DataFrame
        df_combined = pd.DataFrame({
            "SAEV Service %"            : service_percentage        ,
            "New Demand"                : new_demand                ,
            "Fulfilled"                 : fulfilled                 ,
            "Remaining Unfulfilled"     : unfulfilled               ,
            "Cumulative Expired Demand" : cumulative_expired_demand ,
        }, index=TIMESTEPS)
        df_combined.index.name = "Time Interval"
        
        # Create dual-axis plot
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Left axis - SAEV Service percentage
        color = 'tab:blue'
        ax1.set_xlabel('Time Intervals')
        ax1.set_ylabel('% of SAEVs in Service', color=color)
        line1 = ax1.plot(
            df_combined.index           , 
            df_combined["SAEV Service %"] , 
            color       = color         , 
            linewidth   = 2             , 
            marker      = markers[0]    ,
            label       = 'SAEV Service % (left y-axis)' ,
        )
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        # Right axis - Demand metrics
        ax2 = ax1.twinx()
        ax2.set_ylabel('Number of Rides', color='tab:red')
        line2 = ax2.plot(
            df_combined.index           ,  
            df_combined["New Demand"]   , 
            color       = 'tab:purple'  , 
            linewidth   = 2             , 
            marker      = markers[1]    ,
            label       = 'New Demand (right y-axis)' , 
            linestyle   = '--'          ,
        )
        line3 = ax2.plot(
            df_combined.index           ,
            df_combined["Fulfilled"]    ,
            color       = 'tab:green'  ,
            linewidth   = 2             ,
            marker      = markers[4]    ,
            label       = 'Fulfilled (right y-axis)'   ,
            linestyle   = '--'          ,
        )
        line4 = ax2.plot(
            df_combined.index                       , 
            df_combined["Remaining Unfulfilled"]    , 
            color       = 'tab:orange'              , 
            linewidth   = 2                         , 
            marker      = markers[2]                ,
            label       = 'Remaining Unfulfilled (right y-axis)'   , 
            linestyle   = '--'                      ,
        )
        line5 = ax2.plot(
            df_combined.index                           , 
            df_combined["Cumulative Expired Demand"]    , 
            color       = 'tab:red'                     , 
            linewidth   = 2                             , 
            marker      = markers[3]                    ,
            label       = 'Cumulative Expired Demand (right y-axis)'   , 
            linestyle   = '--'                          ,
        )
        
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        # Combined legend
        lines = line1 + line2 + line3 + line4 + line5
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best', frameon=False)
        
        plt.title('SAEV Service Operations vs Demand Metrics')
        plt.tight_layout()
        
        outfile = os.path.join ("Results", folder_name, f"service_demand_comparison_{file_name}_{timestamp}.png")
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved Service and Demand Comparison plot to {outfile}")
        
        return df_combined

    def _electricity_usage_and_pricing():
        """
        Create a plot comparing electricity usage with pricing.
        Left y-axis: SoC levels (usage, max supply, threshold)
        Right y-axis: electricity price (high, low, wholesale)
        
        Returns:
            pd.DataFrame: Electricity usage and pricing data
        """
        # Calculate electricity usage by time
        electricity_usage = [0.0] * (T + 1)
        max_supply = [0.0] * (T + 1)
        threshold = [0.0] * (T + 1)
        
        for t in TIMESTEPS:
            # Calculate total electricity used at time t
            for e_id in charge_arcs_t.get(t, set()):
                arc = all_arcs[e_id]
                # Electricity used = number of EVs * charge amount
                charge_amount = arc.d.l - arc.o.l  # SoC levels charged
                electricity_usage[t] += x[e_id] * charge_amount
            
            # Calculate max supply and threshold for all zones at time t
            for i in ZONES:
                max_supply[t] += elec_supplied.get((i, t), 0.0)
            threshold[t] = elec_threshold.get(t, 0.0)
        
        usage_mean = sum(electricity_usage[1:T]) / (T - 1)
        usage_var = sum((u - usage_mean) ** 2 for u in electricity_usage[1:T]) / (T - 1)
        logger.info(f"Electricity Usage - Mean: {usage_mean:.2f}, Variance: {usage_var:.2f}")
        
        # Create DataFrame
        # Build wholesale price vector
        wholesale_price = [wholesale_elec_price[t] if t in wholesale_elec_price else 0.0 for t in TIMESTEPS]

        df_electricity = pd.DataFrame({
            "Electricity Usage (SoC)"   : electricity_usage                                 ,
            "Max Supply (SoC)"          : max_supply                                        ,
            "Threshold (SoC)"           : threshold                                         ,
            "Price High ($/SoC)"        : [charge_cost_high.get(t, 0.0) for t in TIMESTEPS] ,
            "Price Low ($/SoC)"         : [charge_cost_low.get(t, 0.0) for t in TIMESTEPS]  ,
            "Wholesale Price ($/SoC)"   : wholesale_price                                   ,
        }, index=TIMESTEPS)
        df_electricity.index.name = "Time Interval"
        
        # Create dual-axis plot
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Left axis - SoC levels
        ax1.set_xlabel('Time Intervals')
        ax1.set_ylabel('SoC Levels', color='tab:blue')
        line1 = ax1.plot(
            df_electricity.index                        , 
            df_electricity["Electricity Usage (SoC)"]   , 
            color       = 'tab:blue'                    , 
            linewidth   = 2                             , 
            marker      = markers[0]                    ,
            label       = 'Electricity Usage (left y-axis)' ,
        )
        line2 = ax1.plot(
            df_electricity.index                , 
            df_electricity["Max Supply (SoC)"]  , 
            color       = 'tab:red'             , 
            linewidth   = 2                     , 
            marker      = markers[1]            ,            
            label       = 'Max Supply (left y-axis)'          , 
            linestyle   = '--'                  ,
        )
        line3 = ax1.plot(
            df_electricity.index                , 
            df_electricity["Threshold (SoC)"]   , 
            color       = 'tab:orange'          , 
            linewidth   = 2                     , 
            marker      = markers[2]            ,
            label       = 'Threshold (left y-axis)'           , 
            linestyle   = '--'                  ,
        )
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, alpha=0.3)
        
        # Right axis - Prices
        ax2 = ax1.twinx()
        ax2.set_ylabel('Price ($/SoC)', color='tab:red')
        line4 = ax2.plot(
            df_electricity.index                , 
            df_electricity["Price High ($/SoC)"], 
            color       = 'xkcd:light red'      ,        
            linewidth   = 2                     , 
            marker      = markers[3]            ,
            label       = 'Price High (right y-axis)'          , 
            linestyle   = ':'                   ,
        )
        line5 = ax2.plot(
            df_electricity.index                , 
            df_electricity["Price Low ($/SoC)"] , 
            color       = 'xkcd:light green'    ,     
            linewidth   = 2                     , 
            marker      = markers[4]            ,
            label       = 'Price Low (right y-axis)'           , 
            linestyle   = ':'                   ,
        )
        line6 = ax2.plot(
            df_electricity.index                    , 
            df_electricity["Wholesale Price ($/SoC)"] , 
            color       = 'tab:gray'                 , 
            linewidth   = 2                          , 
            marker      = markers[5]                 ,
            label       = 'Wholesale Price (right y-axis)'          , 
            linestyle   = ':'                        ,
        )
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        # Combined legend
        lines = line1 + line2 + line3 + line4 + line5 + line6
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best', frameon=False)
        
        plt.title('Electricity Usage and Pricing Over Time')
        plt.tight_layout()
        
        outfile = os.path.join ("Results", folder_name, f"electricity_usage_pricing_{file_name}_{timestamp}.png")
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved Electricity Usage and Pricing plot to {outfile}")
        
        return df_electricity

    def _electricity_by_zone():
        """
        Create a DataFrame with electricity usage and max supply broken down by zone.
        Rows: time intervals
        Columns: MultiIndex with zones as main headers and metrics as subheaders
        
        Returns:
            pd.DataFrame: Electricity metrics by zone and time
        """
        # Initialize data structure to collect metrics
        electricity_data = [{} for _ in range(T + 1)]
        
        # Calculate electricity metrics by zone and time
        for t in TIMESTEPS:
            for zone in ZONES:
                # Calculate electricity used in this zone at time t
                usage = 0.0
                num_charging = 0.0
                for e_id in charge_arcs_it.get((zone, t), set()):
                    arc = all_arcs[e_id]
                    charge_amount = arc.d.l - arc.o.l  # SoC levels charged
                    usage += x[e_id] * charge_amount
                    num_charging += x[e_id]
                
                # Store metrics for this zone at time t
                electricity_data[t][zone] = {
                    "Usage (SoC)"       : usage,
                    "Max Supply (SoC)"  : elec_supplied.get((zone, t), 0.0),
                    "EVs Charging"      : num_charging,
                    "Num Ports"         : num_ports.get(zone, 0),
                }
        
        # Convert to DataFrame with MultiIndex columns
        metrics = ["Usage (SoC)", "Max Supply (SoC)", "EVs Charging", "Num Ports"]
        
        # Build column MultiIndex: zones and metrics
        cols = pd.MultiIndex.from_product(
            [ZONES, metrics], 
            names=["Zone", "Metric"]
        )
        
        # Index: all time intervals
        index = TIMESTEPS
        
        # Initialize DataFrame
        df = pd.DataFrame(index=index, columns=cols, dtype=float)
        df.index.name = "Time Interval"
        
        # Fill values from electricity_data
        for t in TIMESTEPS:
            for zone in ZONES:
                zone_dict = electricity_data[t].get(zone, {})
                for metric in metrics:
                    df.loc[t, (zone, metric)] = zone_dict.get(metric, 0)
        
        # Optionally add a total row
        total_row_data = {}
        for zone in ZONES:
            total_usage = sum(
                electricity_data[t].get(zone, {}).get("Usage (SoC)", 0)
                for t in TIMESTEPS
            )
            # For max supply, we might want the sum or the max across time
            # Using sum here to show total capacity over all time
            total_max_supply = sum(
                electricity_data[t].get(zone, {}).get("Max Supply (SoC)", 0)
                for t in TIMESTEPS
            )
            total_evs = sum(
                electricity_data[t].get(zone, {}).get("EVs Charging", 0)
                for t in TIMESTEPS
            )
            
            total_row_data[(zone, "Usage (SoC)")] = total_usage
            total_row_data[(zone, "Max Supply (SoC)")] = total_max_supply
            total_row_data[(zone, "EVs Charging")] = total_evs
        
        # Add total row
        df.loc["total"] = total_row_data
        
        return df
    
    def _plot_zone_saturation(
        metric_name: str,
        title_suffix: str,
        filename_suffix: str,
    ):
        """
        Generates a stacked area plot showing the % of zones exceeding specific utilization thresholds
        (10%, 30%, 50%, 70%, 90%) over time.
        
        Also returns a DataFrame with the raw saturation data and Top/Bottom 3 zones.
        """
        
        # Thresholds to visualize (High saturation to Low saturation)
        thresholds = [0.9, 0.7, 0.5, 0.3, 0.1]
        
        # Colors: Red (High Saturation) -> Green (Low Saturation)
        colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60']
        labels = [f"≥ {int(t*100)}%" for t in thresholds]
        
        # Data storage
        saturation_counts = {t: [] for t in thresholds}
        detailed_rows = []

        # ---------------------------------------------------------
        # FIX: Define indices strictly as 0 to T-1 (excluding T)
        # This ensures X and Y axes always have the same length (24 vs 24)
        # ---------------------------------------------------------
        plot_indices = TIMESTEPS[:-1] 

        # Iterate only through the active time steps
        for t in plot_indices:
            zone_stats = [] # Tuple (zone_id, saturation_ratio)

            # 1. Calculate saturation for every zone at this time step (excluding zero-capacity zones)
            for zone in ZONES:
                usage = 0.0
                capacity = 0.0

                if metric_name == "Usage (SoC)":
                    # Sum electricity usage
                    for e_id in charge_arcs_it.get((zone, t), set()):
                        arc = all_arcs[e_id]
                        usage += x[e_id] * (arc.d.l - arc.o.l)
                    capacity = elec_supplied.get((zone, t), 0.0)
                
                elif metric_name == "EVs Charging":
                    # Sum active charging arcs
                    usage = sum(x[e_id] for e_id in charge_arcs_it.get((zone, t), set()))
                    capacity = num_ports.get(zone, 0)

                # Skip zones with zero capacity
                if capacity == 0:
                    continue
                
                # Calculate saturation ratio
                ratio = (usage / capacity) if capacity > 0 else 0.0
                zone_stats.append((zone, ratio))

            # 2. Count how many zones pass each threshold (only considering non-zero capacity zones)
            total_zones = len(zone_stats)
            for thresh in thresholds:
                count = sum(1 for z, r in zone_stats if r >= thresh - 0.001) # tolerance
                pct = (count / total_zones) * 100 if total_zones > 0 else 0.0
                saturation_counts[thresh].append(pct)

            # 3. Sort for Top 3 / Bottom 3
            zone_stats.sort(key=lambda item: item[1], reverse=True)
            
            top_3 = zone_stats[:3]
            # Bottom 3 (raw)
            bottom_3 = zone_stats[-3:] if len(zone_stats) >= 3 else zone_stats

            # Prepare row for Excel
            row_dict = {}
            for thresh in thresholds:
                row_dict[f"% Zones ≥ {int(thresh*100)}%"] = saturation_counts[thresh][-1]
            
            # Add Top 3
            for rank, (z, r) in enumerate(top_3):
                row_dict[f"Top {rank+1} Zone Index"] = z
                row_dict[f"Top {rank+1} Zone % Saturation"] = round(r * 100, 1)
            
            # Add Bottom 3
            for rank, (z, r) in enumerate(bottom_3):
                true_rank = len(bottom_3) - rank 
                row_dict[f"Bottom {true_rank} Zone Index"] = z
                row_dict[f"Bottom {true_rank} Zone % Saturation"] = round(r * 100, 1)

            detailed_rows.append(row_dict)

        # -----------------------------
        # PLOTTING
        # -----------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Reverse order for plotting (Largest area first = 10% threshold)
        check_thresholds = list(reversed(thresholds)) 
        check_colors = list(reversed(colors))
        check_labels = list(reversed(labels))

        for i, thresh in enumerate(check_thresholds):
            y_values = saturation_counts[thresh]
            
            # Validate lengths before plotting to catch future errors
            if len(plot_indices) != len(y_values):
                logger.error(f"Shape mismatch: X ({len(plot_indices)}) vs Y ({len(y_values)})")
                continue

            ax.fill_between(
                plot_indices, 
                y_values, 
                0, 
                color=check_colors[i], 
                label=check_labels[i], 
                alpha=0.8
            )
            ax.plot(plot_indices, y_values, color='black', linewidth=0.5, alpha=0.3)

        ax.set_xlabel("Time Step")
        ax.set_ylabel("% of Zones")
        ax.set_title(f"% Zone Saturation by {title_suffix} (Stacked)")
        ax.set_ylim(0, 105)
        # x-axis limit matches the indices we plotted
        ax.set_xlim(0, max(plot_indices) if plot_indices else 0)
        ax.grid(True, alpha=0.3)
        
        handles, leg_labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(leg_labels), loc='center left', bbox_to_anchor=(1, 0.5), title=title_suffix)
        
        plt.tight_layout()
        outfile = os.path.join("Results", folder_name, f"{filename_suffix}_{file_name}_{timestamp}.png")
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {title_suffix} saturation plot to {outfile}")

        # -----------------------------
        # EXCEL DATAFRAME
        # -----------------------------
        df_saturation = pd.DataFrame(detailed_rows, index=plot_indices)
        df_saturation.index.name = "Time Step"
        return df_saturation
    
    def _round_df(df):
        """
        Round numeric values in a DataFrame to 3 decimal places.
        Returns a new DataFrame with numeric columns rounded.
        """
        try:
            return df.round(3)
        except Exception:
            # Fallback: round numeric columns in-place
            numeric_cols = df.select_dtypes(include=["number"]).columns
            for col in numeric_cols:
                df[col] = df[col].round(3)
            return df
    
    # Create an excel in the Results folder to save results
    with pd.ExcelWriter(results_name, engine="openpyxl") as writer:
        # 1. Fleet operator profit summary
        df_summary = _fleet_operator_profit()
        df_summary = _round_df(df_summary)
        df_summary.to_excel(writer, sheet_name="Fleet Operator Profit", index_label="Time Step")
        logger.info("1. Saved Fleet Operator Profit to excel")

        # 2. EV operations over time
        df_ev_operations = _ev_operations_over_time()
        df_ev_operations = _round_df(df_ev_operations)
        df_ev_operations.to_excel(writer, sheet_name="EV Operations", index_label="Time Step")
        logger.info("2. Saved EV Operations to excel")

        # 3. Demand served over time
        df_demand_served = _demand_served_over_time()
        # Round values before saving
        df_demand_served = _round_df(df_demand_served)
        if df_demand_served.shape[1] > 10000:
            logger.warning("Demand Served DataFrame has more than 10,000 columns. Saving only final \"total\" columns in Excel.")
            df_demand_served["total"].to_excel(writer, sheet_name="Demand Served", index_label="Time Step")
            logger.warning("Saving full detailed version as CSV")
            df_demand_served.to_csv(results_name.replace(".xlsx", "_demand_served_full.csv"), index_label="Time Step")
        else:
            df_demand_served.to_excel(writer, sheet_name="Demand Served", index_label="Time Step")
        logger.info("3. Saved Demand Served to excel")

        # 4. Service and demand comparison
        df_service_demand = _service_and_demand_comparison()
        df_service_demand = _round_df(df_service_demand)
        df_service_demand.to_excel(writer, sheet_name="Service vs Demand", index_label="Time Step")
        logger.info("4. Saved Service and Demand Comparison to excel")

        # 5. Electricity usage and pricing
        df_electricity = _electricity_usage_and_pricing()
        df_electricity = _round_df(df_electricity)
        df_electricity.to_excel(writer, sheet_name="Electricity Usage vs Pricing", index_label="Time Step")
        logger.info("5. Saved electricity usage and pricing to excel")

        # 6. Electricity by zone
        df_zone_electricity = _electricity_by_zone()
        df_zone_electricity = _round_df(df_zone_electricity)
        df_zone_electricity.to_excel(writer, sheet_name="Electricity by Zone", index_label="Time Step")
        logger.info("6. Saved electricity by zone to excel")

        # 7. Zone saturation plots and data - EV Charging
        df_elec_sat = _plot_zone_saturation(
            metric_name     = "Usage (SoC)",
            title_suffix    = "Electricity Utilization Rate",
            filename_suffix = "saturation_elec",
        )
        df_elec_sat = _round_df(df_elec_sat)
        df_elec_sat.to_excel(writer, sheet_name="Saturation - Electricity", index_label="Time Step")
        logger.info("7. Saved Electricity Saturation data to excel")

        # 8. Zone saturation plots and data - EV Ports
        df_port_sat = _plot_zone_saturation(
            metric_name     = "EVs Charging",
            title_suffix    = "Port Utilization Rate",
            filename_suffix = "saturation_port",
        )
        df_port_sat = _round_df(df_port_sat)
        df_port_sat.to_excel(writer, sheet_name="Saturation - Ports", index_label="Time Step")
        logger.info("8. Saved Port Saturation data to excel")

    logger.info(f"All results saved to {results_name}")
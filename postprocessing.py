import pandas as pd
import matplotlib.pyplot as plt

from logger import Logger
from networkClass import Node, Arc, ArcType

def postprocessing(**kwargs):
    # Input parameters
    N                       : int                                       = kwargs.get("N")                       # number of operation zones (1, ..., N)
    T                       : int                                       = kwargs.get("T")                       # termination time of daily operations (0, ..., T)
    L                       : int                                       = kwargs.get("L")                       # max SoC level (all EVs start at this level) (0, ..., L)
    W                       : int                                       = kwargs.get("W")                       # maximum time intervals a passenger will wait for a ride (0, ..., W-1; demand expires at W)
    
    travel_demand           : dict[tuple[int, int, int], int]           = kwargs.get("travel_demand")           # travel demand from zone i to j at starting at time t
    travel_time             : dict[tuple[int, int, int], int]           = kwargs.get("travel_time")             # travel time from i to j at starting at time t
    travel_energy           : dict[tuple[int, int], int]                = kwargs.get("travel_energy")           # energy consumed for trip from zone i to j
    order_revenue           : dict[tuple[int, int, int], float]         = kwargs.get("order_revenue")           # order revenue for each trip served from i to j at time t
    penalty                 : dict[tuple[int, int, int], float]         = kwargs.get("penalty")                 # penalty cost for each unserved trip from i to j at time t 
    L_min                   : int                                       = kwargs.get("L_min")                   # min SoC level all EV must end with at the end of the daily operations  
    num_EVs                 : int                                       = kwargs.get("num_EVs")                 # total number of EVs in the fleet
    
    num_ports               : dict[int, int]                            = kwargs.get("num_ports")               # number of chargers in each zone
    elec_supplied           : dict[tuple[int, int], int]                = kwargs.get("elec_supplied")           # electricity supplied at zone i at time t
    max_charge_speed        : int                                       = kwargs.get("max_charge_speed")        # max charge speed (kWh per time interval)
    wholesale_elec_price    : dict[int, float]                          = kwargs.get("wholesale_elec_price")    # wholesale electricity price at time t ($ per kWh)

    charge_cost_low         : dict[int, float]                          = kwargs.get("charge_cost_low")         # low charge cost at time t ($ per kWh)
    charge_cost_high        : dict[int, float]                          = kwargs.get("charge_cost_high")        # high charge cost at time t ($ per kWh)
    elec_threshold          : dict[int, int]                            = kwargs.get("elec_threshold")          # threshold for low/high charge cost ($ per kWh)
    
    # Metadata
    timestamp               : str                                       = kwargs.get("timestamp", "")           # timestamp for logging
    file_name               : str                                       = kwargs.get("file_name", "")           # filename for logging
    results_name            : str                                       = kwargs.get("results_name", "")        # filename for results
    
    # Output results
    obj                     : float                                     = kwargs["obj"]
    x                       : dict[int, int]                            = kwargs["x"]  
    s                       : dict[tuple[int, int, int], int]           = kwargs["s"]  
    u                       : dict[tuple[int, int, int, int], int]      = kwargs["u"]  
    e                       : dict[tuple[int, int, int], int]           = kwargs["e"]  
    z                       : dict[int, int]                            = kwargs["z"]
    y                       : dict[int, float]                          = kwargs["y"]
    h                       : float                                     = kwargs["h"]
    l                       : float                                     = kwargs["l"]
    total_service_revenue   : float                                     = kwargs["total_service_revenue"]
    total_penalty_cost      : float                                     = kwargs["total_penalty_cost"]
    total_charge_cost       : float                                     = kwargs["total_charge_cost"]

    # Arcs
    all_arcs                : dict[int                  , Arc]          = kwargs["all_arcs"]
    type_arcs               : dict[ArcType              , set[int]]     = kwargs["type_arcs"]
    in_arcs                 : dict[Node                 , set[int]]     = kwargs["in_arcs"]
    out_arcs                : dict[Node                 , set[int]]     = kwargs["out_arcs"]
    service_arcs_ijt        : dict[tuple[int, int, int] , set[int]]     = kwargs["service_arcs_ijt"]
    charge_arcs_it          : dict[tuple[int, int]      , set[int]]     = kwargs["charge_arcs_it"]
    charge_arcs_t           : dict[int                  , set[int]]     = kwargs["charge_arcs_t"]

    # Sets
    valid_travel_demand     : dict[tuple[int, int, int], int]           = kwargs["valid_travel_demand"]
    invalid_travel_demand   : set[tuple[int, int, int]]                 = kwargs["invalid_travel_demand"]
    ZONES                   : list[int]                                 = kwargs["ZONES"]
    TIMESTEPS               : list[int]                                 = kwargs["TIMESTEPS"]
    LEVELS                  : list[int]                                 = kwargs["LEVELS"]
    AGES                    : list[int]                                 = kwargs["AGES"]
    
    logger = Logger("postprocessing", level="DEBUG", to_console=True, timestamp=timestamp)
    logger.save("postprocessing_" + file_name) 
    logger.info("Parameters loaded successfully")
    
    def _bluebird_profit():
        """
        Build a DataFrame with profit summary:
        - Total Profit
        - Total Service Revenue
        - Total Penalty Cost
        - Total Charging Cost
        
        Returns:
            pd.DataFrame: Summary of profit breakdown
        """
        summary_data = [
            {"description": "Total Profit", "value": total_service_revenue - total_penalty_cost - total_charge_cost},
            {"description": "Total Service Revenue", "value": total_service_revenue},
            {"description": "Total Penalty Cost", "value": total_penalty_cost},
            {"description": "Total Charging Cost", "value": total_charge_cost},
        ]
        df_summary = pd.DataFrame(summary_data, columns=["description", "value"])
        return df_summary

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
        for col in df_ev_operations.columns:
            label = col.split("Percentage of ")[1].split(" EVs")[0]  # Extract arc type name
            plt.plot(df_ev_operations.index, df_ev_operations[col], marker="", linewidth=2, label=label)

        plt.xlabel("Time Intervals")
        plt.ylabel("% of EVs")
        plt.title("EV Operations by ArcType over Time")
        plt.grid(True, alpha=0.3)
        plt.legend(title="ArcType", loc="best", frameon=False)
        plt.tight_layout()

        outfile = f"Results/ev_operations_over_time_{file_name}_{timestamp}.png"
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved EV operations over time plot to {outfile}")

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
                "new demand":                   total_new_demand_by_time,
                "served demand":                total_served_by_time,
                "remaining unserved demand":    total_unserved_by_time,
                "expired demand":               total_expired_by_time
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
        Create a plot comparing EV service operations with demand metrics.
        Left y-axis: percentage of EVs in service
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
        new_demand = []
        unfulfilled = []
        cumulative_expired_demand = []
        
        for t in TIMESTEPS:
            new_demand.append(demand_metrics[t]["total"]["new demand"])
            cumulative_expired_demand.append(demand_metrics[t]["total"]["expired demand"] + (cumulative_expired_demand[t-1] if t > 0 else 0))
            unfulfilled.append(demand_metrics[t]["total"]["remaining unserved demand"])
        
        # Create DataFrame
        df_combined = pd.DataFrame({
            "EV Service %": service_percentage,
            "New Demand": new_demand,
            "Remaining Unfulfilled": unfulfilled,
            "Cumulative Expired Demand": cumulative_expired_demand
        }, index=TIMESTEPS)
        df_combined.index.name = "Time Interval"
        
        # Create dual-axis plot
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Left axis - EV Service percentage
        color = 'tab:blue'
        ax1.set_xlabel('Time Intervals')
        ax1.set_ylabel('% of EVs in Service', color=color)
        line1 = ax1.plot(df_combined.index, df_combined["EV Service %"], 
                        color=color, linewidth=2, label='EV Service %', marker='o', markersize=4)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        # Right axis - Demand metrics
        ax2 = ax1.twinx()
        ax2.set_ylabel('Number of Rides', color='tab:red')
        line2 = ax2.plot(df_combined.index, df_combined["New Demand"], 
                        color='tab:green', linewidth=2, label='New Demand', marker='s', markersize=4)
        line3 = ax2.plot(df_combined.index, df_combined["Remaining Unfulfilled"], 
                        color='tab:orange', linewidth=2, label='Remaining Unfulfilled', marker='^', markersize=4)
        line4 = ax2.plot(df_combined.index, df_combined["Cumulative Expired Demand"], 
                        color='tab:red', linewidth=2, label='Cumulative Expired Demand', marker='d', markersize=4)
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        # Combined legend
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best', frameon=False)
        
        plt.title('EV Service Operations vs Demand Metrics')
        plt.tight_layout()
        
        outfile = f"Results/service_demand_comparison_{file_name}_{timestamp}.png"
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved service and demand comparison plot to {outfile}")
        
        return df_combined

    def _electricity_usage_and_pricing():
        """
        Create a plot comparing electricity usage with pricing.
        Left y-axis: SoC levels (usage, max supply, threshold)
        Right y-axis: electricity price (high and low)
        
        Returns:
            pd.DataFrame: Electricity usage and pricing data
        """
        # Calculate electricity usage by time
        electricity_usage = [0] * (T + 1)
        max_supply = [0] * (T + 1)
        threshold = [0] * (T + 1)
        
        for t in TIMESTEPS:
            # Calculate total electricity used at time t
            for e_id in charge_arcs_t.get(t, set()):
                arc = all_arcs[e_id]
                # Electricity used = number of EVs * charge amount
                charge_amount = arc.d.l - arc.o.l  # SoC levels charged
                electricity_usage[t] += x[e_id] * charge_amount
            
            # Calculate max supply and threshold for all zones at time t
            for i in ZONES:
                max_supply[t] += elec_supplied.get((i, t), 0)
            threshold[t] = elec_threshold.get(t, 0)
        
        # Create DataFrame
        df_electricity = pd.DataFrame({
            "Electricity Usage (SoC)": electricity_usage,
            "Max Supply (SoC)": max_supply,
            "Threshold (SoC)": threshold,
            "Price High ($/SoC)": [charge_cost_high.get(t, 0) for t in TIMESTEPS],
            "Price Low ($/SoC)": [charge_cost_low.get(t, 0) for t in TIMESTEPS]
        }, index=TIMESTEPS)
        df_electricity.index.name = "Time Interval"
        
        # Create dual-axis plot
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Left axis - SoC levels
        ax1.set_xlabel('Time Intervals')
        ax1.set_ylabel('SoC Levels', color='tab:blue')
        line1 = ax1.plot(df_electricity.index, df_electricity["Electricity Usage (SoC)"], 
                        color='tab:blue', linewidth=2, label='Electricity Usage', marker='o', markersize=4)
        line2 = ax1.plot(df_electricity.index, df_electricity["Max Supply (SoC)"], 
                        color='tab:cyan', linewidth=2, label='Max Supply', linestyle='--', marker='s', markersize=4)
        line3 = ax1.plot(df_electricity.index, df_electricity["Threshold (SoC)"], 
                        color='tab:purple', linewidth=2, label='Threshold', linestyle=':', marker='^', markersize=4)
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, alpha=0.3)
        
        # Right axis - Prices
        ax2 = ax1.twinx()
        ax2.set_ylabel('Price ($/SoC)', color='tab:red')
        line4 = ax2.plot(df_electricity.index, df_electricity["Price High ($/SoC)"], 
                        color='tab:red', linewidth=2, label='Price High', marker='d', markersize=4)
        line5 = ax2.plot(df_electricity.index, df_electricity["Price Low ($/SoC)"], 
                        color='tab:orange', linewidth=2, label='Price Low', marker='v', markersize=4)
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        # Combined legend
        lines = line1 + line2 + line3 + line4 + line5
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best', frameon=False)
        
        plt.title('Electricity Usage and Pricing Over Time')
        plt.tight_layout()
        
        outfile = f"Results/electricity_usage_pricing_{file_name}_{timestamp}.png"
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved electricity usage and pricing plot to {outfile}")
        
        return df_electricity

    def _electricity_by_zone():
        """
        Create a DataFrame with electricity usage and max supply broken down by zone.
        Rows: time intervals
        Columns: each zone's usage and max supply
        
        Returns:
            pd.DataFrame: Electricity metrics by zone and time
        """
        # Initialize data structure
        data = {}
        for zone in ZONES:
            data[f"Zone {zone} Usage (SoC)"] = [0] * (T + 1)
            data[f"Zone {zone} Max Supply (SoC)"] = [0] * (T + 1)
        
        # Calculate electricity usage by zone and time
        for t in TIMESTEPS:
            for zone in ZONES:
                # Calculate electricity used in this zone at time t
                for e_id in charge_arcs_it.get((zone, t), set()):
                    arc = all_arcs[e_id]
                    charge_amount = arc.d.l - arc.o.l  # SoC levels charged
                    data[f"Zone {zone} Usage (SoC)"][t] += x[e_id] * charge_amount
                
                # Max supply for this zone at time t
                data[f"Zone {zone} Max Supply (SoC)"][t] = elec_supplied.get((zone, t), 0)
        
        # Create DataFrame
        df_zone_electricity = pd.DataFrame(data, index=TIMESTEPS)
        df_zone_electricity.index.name = "Time Interval"
        
        return df_zone_electricity

    # Create an excel in the Results folder to save results
    with pd.ExcelWriter(results_name, engine="openpyxl") as writer:
        # 1. Bluebird profit summary
        df_summary = _bluebird_profit()
        df_summary.to_excel(writer, sheet_name="Bluebird Profit Summary", index=False)
        logger.info("Saved Bluebird profit summary to excel")

        # 2. EV operations over time
        df_ev_operations = _ev_operations_over_time()
        df_ev_operations.to_excel(writer, sheet_name="EV Operations Over Time", index_label="Time Step")
        logger.info("Saved EV operations over time to excel")

        # 3. Demand served over time
        df_demand_served = _demand_served_over_time()
        df_demand_served.to_excel(writer, sheet_name="Demand Served Over Time", index_label="Time Step")
        logger.info("Saved demand served over time to excel")

        # 4. Service and demand comparison
        df_service_demand = _service_and_demand_comparison()
        df_service_demand.to_excel(writer, sheet_name="Service vs Demand", index_label="Time Step")
        logger.info("Saved service and demand comparison to excel")

        # 6. Electricity usage and pricing
        df_electricity = _electricity_usage_and_pricing()
        df_electricity.to_excel(writer, sheet_name="Electricity Usage & Pricing", index_label="Time Step")
        logger.info("Saved electricity usage and pricing to excel")

        # 7. Electricity by zone
        df_zone_electricity = _electricity_by_zone()
        df_zone_electricity.to_excel(writer, sheet_name="Electricity by Zone", index_label="Time Step")
        logger.info("Saved electricity by zone to excel")

    logger.info(f"All results saved to {results_name}")
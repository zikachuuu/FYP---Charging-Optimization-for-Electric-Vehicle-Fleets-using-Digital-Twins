import pandas as pd
import matplotlib.pyplot as plt

from logger import Logger
from networkClass import Node, Arc, ArcType

def postprocessing (
        **kwargs
    ):
    # Input parameters
    N                       : int                                       = kwargs.get("N")               # number of operation zones (1, ..., N)
    T                       : int                                       = kwargs.get("T")               # termination time of daily operations (0, ..., T)
    L                       : int                                       = kwargs.get("L")               # max SoC level (all EVs start at this level) (0, ..., L)
    W                       : int                                       = kwargs.get("W")               # maximum time intervals a passenger will wait for a ride (0, ..., W-1; demand expires at W)
    travel_demand           : dict[tuple[int, int, int], int]           = kwargs.get("travel_demand")   # travel demand from zone i to j at starting at time t
    travel_time             : dict[tuple[int, int, int], int]           = kwargs.get("travel_time")     # travel time from i to j at starting at time t
    travel_energy           : dict[tuple[int, int], int]                = kwargs.get("travel_energy")   # energy consumed for trip from zone i to j
    order_revenue           : dict[tuple[int, int, int], float]         = kwargs.get("order_revenue")   # order revenue for each trip served from i to j at time t
    penalty                 : dict[tuple[int, int, int], float]         = kwargs.get("penalty")         # penalty cost for each unserved trip from i to j at time t 
    charge_speed            : int                                       = kwargs.get("charge_speed")    # charge speed (SoC levels per timestep)
    L_min                   : int                                       = kwargs.get("L_min")           # min SoC level all EV must end with at the end of the daily operations  
    num_ports               : dict[int, int]                            = kwargs.get("num_ports")       # number of chargers in each zone
    num_EVs                 : int                                       = kwargs.get("num_EVs")         # total number of EVs in the fleet
    charge_cost             : dict[int, float]                          = kwargs.get("charge_cost")     # price to charge one SoC level at time t

    # Metadata
    timestamp               : str                                       = kwargs.get("timestamp", "")   # timestamp for logging
    file_name               : str                                       = kwargs.get("file_name", "")   # filename for logging
    results_name            : str                                       = kwargs.get("results_name", "")# filename for results
    
    # Output results
    obj                     : float                                     = kwargs["obj"]
    x                       : dict[int, int]                            = kwargs["x"]  
    s                       : dict[tuple[int, int, int], int]           = kwargs["s"]  
    u                       : dict[tuple[int, int, int, int], int]      = kwargs["u"]  
    e                       : dict[tuple[int, int, int], int]           = kwargs["e"]  
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
        summary_data = [
            {"description": "Total Profit", "value": total_service_revenue - total_penalty_cost - total_charge_cost},
            {"description": "Total Service Revenue", "value": total_service_revenue},
            {"description": "Total Penalty Cost", "value": total_penalty_cost},
            {"description": "Total Charging Cost", "value": total_charge_cost},
        ]
        df_summary = pd.DataFrame(summary_data, columns=["description", "value"])
        return df_summary

    def _demand_served_to_dataframe(demand_served):
        """
        Build a DataFrame with:
        - Index: t in TIMESTEPS plus one final row 'total'
        - Columns: MultiIndex [(i,j), metric] for all i!=j plus ('total', metric)
        """
        # Determine all (i, j) pairs and metrics
        ij_pairs = [(i, j) for i in ZONES for j in ZONES if i != j]
        # Metrics (order as you prefer)
        base_metrics = [
            "new demand",
            "served demand",
            "remaining unserved demand",
            "expired demand",
        ]
        age_metrics = [f"unserved demand of age {age}" for age in AGES]
        metrics = base_metrics + age_metrics

        # Build column MultiIndex: all (i,j) pairs and "total"
        main_cols = ij_pairs + ["total"]
        cols = pd.MultiIndex.from_product([main_cols, metrics], names=["(Start Zone, End Zone)", "metric"])

        # Index: all t plus final 'total'
        index = list(TIMESTEPS) + ["total"]

        # Initialize DataFrame
        df = pd.DataFrame(index=index, columns=cols, dtype=float)

        # Fill values from demand_served list/dict structure
        for t in TIMESTEPS:
            row_key = t
            row_dict = demand_served[t]  # dict with keys (i,j) and "total"
            for od in main_cols:
                od_dict = row_dict.get(od, {})
                for m in metrics:
                    df.loc[row_key, (od, m)] = od_dict.get(m, 0)

        # Final total row from demand_served[T+1]
        total_row = demand_served[len(TIMESTEPS) + 0] if isinstance(TIMESTEPS, range) else demand_served[max(TIMESTEPS) + 1]
        # The above line may be ambiguous depending on TIMESTEPS type.
        # More robust: access by position using T:
        # If you have T available, you can just do: total_row = demand_served[T + 1]
        # Here we'll try to infer T from the demand_served length:
        total_row = demand_served[len(demand_served) - 1]
        for od in main_cols:
            od_dict = total_row.get(od, {})
            for m in metrics:
                df.loc["total", (od, m)] = od_dict.get(m, 0)

        return df

    def _demand_served_over_time():
        demand_served = [{} for _ in range(T + 2)]  # Demand served at each time t, from t=0 to t=T, last row for total
        for t in TIMESTEPS:
            for i in ZONES:
                for j in ZONES:
                    if i == j:
                        continue

                    new_demand      : int  = valid_travel_demand.get((i, j, t), 0)
                    served_demand   : int = sum(x[e_id] for e_id in service_arcs_ijt.get((i, j, t), set()))
                    unserved_demand : int = s[(i, j, t)]
                    expired_demand  : int = e[(i, j, t)]

                    demand_served[t][(i, j)] = {
                        "new demand"                : new_demand,
                        "served demand"             : served_demand,
                        "remaining unserved demand" : unserved_demand,
                        "expired demand"            : expired_demand
                    }

                    for age in AGES:
                        demand_served[t][(i, j)][f"unserved demand of age {age}"] = u[(i, j, t, age)]
            
            total_new_demand_by_time = sum (
                valid_travel_demand.get ((i, j, t), 0)
                for i in ZONES 
                for j in ZONES
            )
            total_served_by_time = sum ( 
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
            total_expired_by_time = sum (
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
        
        total_new_demand = sum(
            valid_travel_demand.values()
        )
        total_served = sum(
            x.get(arc_id, 0) 
            for arc_id in type_arcs[ArcType.SERVICE]
        )
        total_unserved = 0
        total_expired = sum (
            e[(i, j, t)]
            for i in ZONES
            for j in ZONES
            for t in TIMESTEPS
        ) + sum (
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
        
        df_demand_served = _demand_served_to_dataframe(demand_served)
        return df_demand_served
    
    def _ev_operations_over_time():
        """Plot EV operations by ArcType over time."""
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
            label = getattr(col, "name", None) or str(col)  # nicer legend for Enums
            plt.plot(df_ev_operations.index, df_ev_operations[col], marker="", linewidth=2, label=label)

        plt.xlabel("Time Intervals")
        plt.ylabel("% of EVs")
        plt.title("EV Operations by ArcType over Time")
        plt.grid(True, alpha=0.3)
        plt.legend(title="ArcType", loc="best", frameon=False)
        plt.tight_layout()

        outfile = f"Results/ev_operations_over_time_{file_name}_{timestamp}.png"
        # Save and close
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved EV operations over time plot to {outfile}")

        return df_ev_operations
    
    def _ports_utilization_over_time():
        """Plot Charging Ports Utilization, as well price of electricty, over time, on the same graph."""
        total_ports = sum(num_ports.values())
        ports_utilization = [0] * (T + 1)
        for e_id in type_arcs[ArcType.CHARGE]:
            arc = all_arcs[e_id]
            start_t = arc.o.t
            end_t = arc.d.t
            for t in range(start_t, end_t):
                ports_utilization[t] += x[e_id]

        df_ports_utilization = pd.DataFrame({"Charging Ports Utilization": ports_utilization}, index=TIMESTEPS)
        df_ports_utilization = df_ports_utilization.div(total_ports).mul(100)  # convert to percentage
        df_ports_utilization.index.name = "Time Interval"
        df_ports_utilization.columns = ["Percentage of Charging Ports Utilized"]

        df_price = pd.DataFrame({"Electricity Price": [charge_cost[t] for t in TIMESTEPS]}, index=TIMESTEPS)
        df_price.index.name = "Time Interval"
        df_price.columns = ["Electricity Price (per kWh)"]
        ax1 = df_ports_utilization.plot(
            figsize=(10, 5), 
            color="tab:blue", 
            marker="", 
            linewidth=2, 
            legend=False
        )
        ax2 = ax1.twinx()
        df_price.plot(
            ax=ax2, 
            color="tab:orange", 
            marker="", 
            linewidth=2, 
            legend=False
        )
        ax1.set_xlabel("Time Intervals")
        ax1.set_ylabel("% of Charging Ports Utilized", color="tab:blue")
        ax2.set_ylabel("Electricity Price (per kWh)", color="tab:orange")
        plt.title("Charging Ports Utilization and Electricity Price over Time")
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, title="Metrics", loc="best", frameon=False)
        outfile = f"Results/ports_utilization_over_time_{file_name}_{timestamp}.png"
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved Charging Ports Utilization and Electricity Price over time plot to {outfile}")
        return df_ports_utilization, df_price

    # Create an excel in the Results folder to save results
    with pd.ExcelWriter(results_name, engine="openpyxl") as writer:
        df_summary = _bluebird_profit()
        df_summary.to_excel(writer, sheet_name="Bluebird Profit Summary", index=False)
        logger.info("Saved Bluebird profit summary to excel")

        df_ev_operations = _ev_operations_over_time()
        df_ev_operations.to_excel(writer, sheet_name="EV Operations Over Time", index_label="Time Step")
        logger.info("Saved EV operations over time to excel")

        df_demand_served = _demand_served_over_time()
        df_demand_served.to_excel(writer, sheet_name="Demand Served Over Time", index_label="Time Step")
        logger.info("Saved demand served over time to excel")

        df_ports_utilization, df_price = _ports_utilization_over_time()
        # Write both dataframes to the same sheet, first column time, second column ports utilization, third column price
        df_ports_utilization.to_excel(writer, sheet_name="Ports Utilization & Price", index_label="Time Step", startcol=0)
        # Write df_price without its index (time column) to avoid duplicate time columns
        df_price.reset_index(drop=True).to_excel(writer, sheet_name="Ports Utilization & Price", index=False, startcol=2, header=True)
        logger.info("Saved ports utilization and electricity price over time to excel")

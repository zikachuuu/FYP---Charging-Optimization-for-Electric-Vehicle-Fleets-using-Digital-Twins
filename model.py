from __future__ import annotations
import gurobipy as gp
from gurobipy import GRB
from networkClass import Node, Arc, ArcType
from dotenv import load_dotenv
import os

load_dotenv()

from logger import Logger

logger = Logger("model", level="DEBUG", to_console=True)
logger.save("model", mode="a") 

# ----------------------------
# Model builder
# ----------------------------

def model(
        **kwargs
    ) -> dict:
    """
    Builds the SAEV model based on the provided parameters.
    """
    logger.info("Model building started")
    
    # ----------------------------
    # Parameters
    # ----------------------------
    N               : int                               = kwargs.get("N")               # number of operation zones
    T               : int                               = kwargs.get("T")               # termination time of daily operations
    L               : int                               = kwargs.get("L")               # max SoC level
    travel_demand   : dict[tuple[int, int, int], int]   = kwargs.get("travel_demand")   # d_ijt = travel demand from zone i to j at time t
    travel_time     : dict[tuple[int, int, int], int]   = kwargs.get("travel_time")     # τ_ijt = travel time from i to j at time t
    travel_energy   : dict[tuple[int, int], int]        = kwargs.get("travel_energy")   # l_ij = energy consumed for trip i->j
    order_revenue   : dict[tuple[int, int, int], float] = kwargs.get("order_revenue")   # p_e = order revenue for service arcs departing (i,t) to j
    penalty         : dict[tuple[int, int, int], float] = kwargs.get("penalty")         # c_ijt = penalty cost for unserved demand
    charge_speed    : int                               = kwargs.get("charge_speed")    # charge speed (SoC levels per timestep)
    
    # Decision variables that are currently set as parameters for simplicity
    num_ports       : dict[int, int]                    = kwargs.get("num_ports")       # number of chargers in each zone
    num_EVs         : int                               = kwargs.get("num_EVs")         # total number of EVs in the fleet
    charge_cost     : dict[int, float]                  = kwargs.get("charge_cost")     # c^c_e = cost for charging arcs

    logger.info("Parameters loaded successfully")

    # --------------------------
    # Sets
    # --------------------------
    ZONES       : list[int] = list (range(N))       # zones 0..N-1
    TIMESTEPS   : list[int] = list (range(T + 1))   # time steps 0..T
    LEVELS      : list[int] = list (range(L + 1))   # SoC levels 0..L

    logger.info ("Sets initialized successfully")

    # ----------------------------
    # Build nodes V
    # ----------------------------
    V: list[Node] = [           \
        Node(i, t, l)           \
            for i in ZONES      \
            for t in TIMESTEPS  \
            for l in LEVELS     \
    ]
    V_set: set[Node] = set(V)

    logger.info(f"Nodes V built with {len(V)} nodes")


    # ----------------------------
    # Build arcs ξ and all subsets
    # ----------------------------
    all_arcs    : dict[int      , Arc]      = {}                                # Map all arcs to their unique ids
    type_arcs   : dict[ArcType  , set[int]] = {type: set() for type in ArcType} # sets of arc ids by type
    in_arcs     : dict[Node     , set[int]] = {v: set() for v in V}             # incoming arc ids per node
    out_arcs    : dict[Node     , set[int]] = {v: set() for v in V}             # outgoing arc ids per node

    type_starting_arcs_l    : dict[tuple[ArcType, int]  , set[int]] = {}    # arc ids starting at (i,0,l) for all i, indexed by (type, l)
    type_ending_arcs_l      : dict[tuple[ArcType, int]  , set[int]] = {}    # arc ids starting at (i,T,l) for all i, indexed by (type, l)
    service_arcs_ijt        : dict[tuple[int, int, int] , set[int]] = {}    # service arcs from node (i,t,l) to (j,t',l') for all l, t',l', indexed by (i,j,t)
    charge_arcs_it          : dict[tuple[int, int]      , set[int]] = {}    # charging arcs from node (i,t,l) for all l, indexed by (i,t)

    # Associate revenue or cost with arcs by kind
    # Keys are arc ids
    arc_revenue     : dict[int, float] = {}     # service revenue
    arc_penalty     : dict[int, float] = {}     # penalty for unserved demand
    arc_charge_cost : dict[int, float] = {}     # charging cost

    next_id = 0
    def _add_arc(
            type    : ArcType       , 
            o       : Node          ,     
            d       : Node          , 
            revenue : float = 0.0   ,
            cost    : float = 0.0   ,
        ) -> int | None:
        """
        Add an arc of a given type from origin node o to destination node d.
        Also registers the arc in all relevant sets and attaches revenue or cost if applicable.

        Returns the arc id if successful, or None if nodes are not valid.
        
        Note: The arc id is auto-incremented from next_id.
        """

        nonlocal next_id    # needed if we want to modify next_id inside this function

        if o not in V_set or d not in V_set:
            return None
        
        e = Arc(next_id, type, o, d)
        next_id += 1

        all_arcs[e.id] = e
        type_arcs[type].add(e.id)
        in_arcs[o].add(e.id)
        out_arcs[d].add(e.id)
        
        # register sets required by constraints
        if (o.t == 0) and (type != ArcType.WRAP):
            type_starting_arcs_l.setdefault((type, o.l), set()).add(e.id)

        if (o.t == T) and (type == ArcType.WRAP):
            type_ending_arcs_l.setdefault((type, o.l), set()).add(e.id)

        if type == ArcType.SERVICE:
            service_arcs_ijt.setdefault((o.i, d.i, o.t), set()).add(e.id)

        if type == ArcType.CHARGE:
            charge_arcs_it.setdefault((o.i, o.t), set()).add(e.id)

        if type == ArcType.SERVICE:
            # attach service revenue and penalty cost
            arc_revenue[e.id]  = revenue
            penalty[e.id]       = cost
        elif type == ArcType.CHARGE:
            # attach charging cost
            arc_charge_cost[e.id]   = cost

        return e.id
    
    valid_demand    : set[tuple[int, int, int]] = set()
    invalid_demand  : set[tuple[int, int, int]] = set()
    
    # Add all arcs
    for i in ZONES:                 # Starting zone
        for j in ZONES:             # Destination zone
            for t in TIMESTEPS:     # Time when starting the trip

                # ---- Add service arcs ξ^s ----
                if (i, j, t) in travel_demand:
                    # We only add service arcs if there is demand

                    demand_ijt: int = travel_demand[(i, j, t)]
                    if demand_ijt <= 0:
                        invalid_demand.add((i, j, t))
                        logger.warning(f"No demand for service arc ({i}, {j}, {t}), skipping")
                        continue  # no demand, skip this arc

                    travel_time_ijt : int = travel_time.get((i, j, t), 0)
                    travel_energy_ij: int = travel_energy.get((i, j), 0)

                    if travel_time_ijt <= 0 or travel_energy_ij <= 0:
                        invalid_demand.add((i, j, t))
                        logger.warning(f"Invalid travel time or energy for service arc ({i}, {j}, {t}), skipping")
                        continue   # no valid travel time or energy, skip this arc

                    if travel_time_ijt + t > T:
                        invalid_demand.add((i, j, t))
                        logger.warning(f"Travel time exceeds total time T for service arc ({i}, {j}, {t}), skipping")
                        continue  # travel time exceeds total time T, skip this arc

                    if travel_energy_ij > L:
                        invalid_demand.add((i, j, t))
                        logger.warning(f"Travel energy exceeds max SoC L for service arc ({i}, {j}, {t}), skipping")
                        continue

                    for l in range(travel_energy_ij, L + 1):
                        # Need at least travel_energy_ij SoC to serve this demand                       
                        o       = Node(i, t, l)
                        d       = Node(j, t + travel_time_ijt, l - travel_energy_ij)
                        revenue = order_revenue.get((i, j, t), 0.0)
                        cost    = penalty.get((i, j, t), 0.0)
                        _add_arc(ArcType.SERVICE, o, d, revenue, cost)

                    valid_demand.add((i, j, t))

                # ---- Add Relocation arcs ξ^r ----
                if i != j and (i, j, t) in travel_time:
                    # Only add relocation arcs if i != j and travel time is defined

                    travel_time_ijt = travel_time[(i, j, t)]
                    travel_energy_ij = travel_energy.get((i, j), 0)

                    if travel_time_ijt <= 0 or travel_energy_ij <= 0 or travel_time_ijt + t > T:
                        continue

                    for l in range(travel_energy_ij, L + 1):
                        # Need at least travel_energy_ij SoC to relocate
                        o = Node(i, t, l)
                        d = Node(j, t + travel_time_ijt, l - travel_energy_ij)
                        _add_arc(ArcType.RELOCATION, o, d)

                # ---- Add Charging arcs ξ^c ----
                if t in charge_cost and t + 1 <= T:
                    # Only add charging arcs if there is a charge cost defined for this time step

                    for l in LEVELS:
                        # Charging arcs can only be added if there is enough room to charge

                        o = Node(i, t, l)
                        d = Node(i, t + 1, min (l + charge_speed, L))  # next level cannot exceed max SoC L

                        # charging cost per EV in one time step = charging cost per level * charge speed (levels charged in one time step)
                        cost = charge_cost.get(t) * charge_speed

                        _add_arc(ArcType.CHARGE, o, d, cost = cost)

                # ---- Add Idle arcs ξ^p ----
                # Idle arcs are added for every node at every time step
                if t + 1 <= T:
                    for l in LEVELS:
                        o = Node(i, t, l)
                        d = Node(i, t + 1, l)
                        _add_arc(ArcType.IDLE, o, d)

                # ---- Add wrap-around arcs ξ^w (end of day t=T to start of next day t=0) ----
                if t == T:
                    for l in LEVELS:
                        o = Node(i, T, l)
                        d = Node(i, 0, l)
                        _add_arc(ArcType.WRAP, o, d)

    # Log any invalid demand that was skipped
    if invalid_demand:
        logger.warning(f"Skipped {len(invalid_demand)} invalid demand entries: {invalid_demand}")
    
    logger.info(f"Arcs built with {len(all_arcs)} arcs")
    # Log the number of arcs by type
    for arc_type, arcs in type_arcs.items() :
        logger.info(f"  {arc_type.name} arcs: {len(arcs)}")
    

    # ----------------------------
    # Model
    # ----------------------------
    with gp.Env() as env, gp.Model(env=env) as model:        
        model.Params.LogFile = "Logs/gurobi_model.log"  # Set log file for Gurobi

        # ----------------------------
        # Decision variables
        # ----------------------------

        # x_e: number of vehicles flow in arc id e
        x: gp.tupledict[int, gp.Var] = model.addVars(
            all_arcs.keys()         ,
            vtype   = GRB.INTEGER   , 
            lb      = 0             , 
            name    = "x"           ,
        )

        # s_ijt: unserved demand integers
        s: gp.tupledict[tuple[int, int, int], gp.Var] = model.addVars(
            valid_demand            ,
            vtype   = GRB.INTEGER   , 
            lb      = 0             , 
            name    = "s"           ,
        )

        logger.info("Decision variables created")
        logger.info(f"  x_e variables: {len(x)}")
        logger.info(f"  s_ijt variables: {len(s)}")

        # ----------------------------
        # Constraints
        # ----------------------------

        # (4) Flow conservation at every node v ∈ V:
        model.addConstrs(
            gp.quicksum(x[e] for e in out_arcs[v]) - gp.quicksum(x[e] for e in in_arcs[v]) == 0
            for v in V_set
        )
        logger.info("Flow conservation constraints (4) added")

        # (5) For each SOC level l, the number of EVs that start with l cannot be lower than the number that end with l
        #     This ensures that the EVs does not use up all the electricty and ends the day without any electricity
        #     Which is detrimental to the next day operations
        model.addConstrs(
            gp.quicksum(
                x[e] 
                    for arctype in ArcType 
                    for e in type_starting_arcs_l.get((arctype, l), set())
            ) <=
            gp.quicksum(
                x[e] 
                    for arctype in ArcType
                    for e in type_ending_arcs_l.get((arctype, l), set())
            )
            for l in LEVELS
        )
        logger.info("SOC level constraints (5) added")

        # (6) Unserved demand propagation for t ∈ [1, T]
        #     We assume unserved passengers will not leave the system and carry over to the next time step
        #     Thus, unserved demand at time t = unserved demand at t-1 + new demand at t - served demand at t
        model.addConstrs(
            s[(i, j, t)] == s[(i, j, t - 1)] + travel_demand.get((i, j, t), 0) - 
                gp.quicksum(
                    x[e] for e in service_arcs_ijt.get((i, j, t), set())
                )
            for (i, j, t) in valid_demand if t > 0
        )
        logger.info("Unserved demand propagation constraints (6) added")

        # (7) Unserved demand at t=0
        model.addConstrs(
            s[(i, j, 0)] == travel_demand.get((i, j, 0), 0) - 
                gp.quicksum(
                    x[e] for e in service_arcs_ijt.get((i, j, 0), set())
                )
            for (i, j) in {(i, j) for (i, j, t) in valid_demand if t == 0}
        )
        logger.info("Unserved demand at t=0 constraints (7) added")

        # (8) Number of EVs charging at each zone at each time period cannot exceed the number of ports available in that zone
        model.addConstrs(
            gp.quicksum(x[e] for e in charge_arcs_it.get((i, t), set())) <= num_ports.get(i, 0)
            for i in ZONES  
            for t in TIMESTEPS
        )
        logger.info("Charging port constraints (8) added")

        # (9) Fleet size equals sum of wrap-around flows
        model.addConstr(
            gp.quicksum(x[e] for e in type_arcs[ArcType.WRAP]) == num_EVs
        )
        logger.info("Fleet size constraint (9) added")


        # ----------------------------
        # Objective
        # ----------------------------

        total_service_revenue   = model.addVar (name = "total_service_revenue")
        total_penalty_cost      = model.addVar (name = "total_penalty_cost")
        total_charge_cost       = model.addVar (name = "total_charge_cost")

        model.addConstr (
            total_service_revenue == gp.quicksum(
                x[e] * arc_revenue.get(e, 0.0)
                for e in type_arcs[ArcType.SERVICE]
            ),
            name = "total_service_revenue"
        )

        model.addConstr (
            total_penalty_cost == gp.quicksum(
                s[(i, j, t)] * float(penalty.get((i, j, t), 0.0))
                for (i, j, t) in valid_demand
            ),
            name = "total_penalty_cost"
        )

        model.addConstr (
            total_charge_cost == gp.quicksum(
                x[e] * arc_charge_cost.get(e, 0.0)
                for e in type_arcs[ArcType.CHARGE]
            ),
            name = "total_charge_cost"
        )

        model.setObjective(
            total_service_revenue - total_penalty_cost - total_charge_cost,
            GRB.MAXIMIZE
        )
        logger.info("Objective function set")

        model.update()  # Apply all changes to the model
        logger.info("Model built successfully")
        logger.info(f"Optimizing model with {model.NumVars} variables and {model.NumConstrs} constraints")

        model.optimize()

        logger.info(f"Optimization completed with status {model.Status}")

        if model.Status != GRB.OPTIMAL:
            logger.error(f"Optimization was not successful. Status: {model.Status}")
            return None
        
        logger.info("Optimization successful.")
        logger.info(f"  Runtime: {model.Runtime:.4f} seconds")
        logger.info(f"  Objective value: {model.ObjVal:.4f}")

        x_sol: dict[int, int]                   = {e: v.X for e, v in x.items() if v.X > 0} # Keys are arc ids
        s_sol: dict[tuple[int, int, int], int]  = {k: v.X for k, v in s.items() if v.X > 0} # Keys are (i, j, t) tuples
        total_service_revenue_sol   : float     = total_service_revenue.X
        total_penalty_cost_sol      : float     = total_penalty_cost.X
        total_charge_cost_sol       : float     = total_charge_cost.X

        return {
            "obj"                       : model.ObjVal,
            "sol": {
                "x"                     : x_sol,
                "s"                     : s_sol,
                "total_service_revenue" : total_service_revenue_sol,
                "total_penalty_cost"    : total_penalty_cost_sol,
                "total_charge_cost"     : total_charge_cost_sol
            },
            "sets": {
                "all_arcs"              : all_arcs,
                "type_arcs"             : type_arcs,
                "in_arcs"               : in_arcs,
                "out_arcs"              : out_arcs,
                "type_starting_arcs_l"  : type_starting_arcs_l,
                "type_ending_arcs_l"    : type_ending_arcs_l,
                "service_arcs_ijt"      : service_arcs_ijt,
                "charge_arcs_it"        : charge_arcs_it
            },
        }


from __future__ import annotations
import gurobipy as gp
from gurobipy import GRB
from networkClass import Node, Arc, ArcType
from dotenv import load_dotenv
import os

load_dotenv()

from logger import Logger

# ----------------------------
# Model builder
# ----------------------------

def model(
        **kwargs
    ) -> dict:
    """
    Builds the SAEV model based on the provided parameters.
    """    
    # ----------------------------
    # Parameters
    # ----------------------------
    N               : int                               = kwargs.get("N")               # number of operation zones (1, ..., N)
    T               : int                               = kwargs.get("T")               # termination time of daily operations (0, ..., T)
    L               : int                               = kwargs.get("L")               # max SoC level (all EVs start at this level) (0, ..., L)
    W               : int                               = kwargs.get("W")               # maximum time intervals a passenger will wait for a ride (0, ..., W-1; demand expires at W)

    travel_demand   : dict[tuple[int, int, int], int]   = kwargs.get("travel_demand")   # travel demand from zone i to j at starting at time t
    travel_time     : dict[tuple[int, int, int], int]   = kwargs.get("travel_time")     # travel time from i to j at starting at time t
    travel_energy   : dict[tuple[int, int], int]        = kwargs.get("travel_energy")   # energy consumed for trip from zone i to j
    order_revenue   : dict[tuple[int, int, int], float] = kwargs.get("order_revenue")   # order revenue for each trip served from i to j at time t
    penalty         : dict[tuple[int, int, int], float] = kwargs.get("penalty")         # penalty cost for each unserved trip from i to j at time t 
    charge_speed    : int                               = kwargs.get("charge_speed")    # charge speed (SoC levels per timestep)
    L_min           : int                               = kwargs.get("L_min")           # min SoC level all EV must end with at the end of the daily operations
    
    # Decision variables that are currently set as parameters for simplicity
    num_ports       : dict[int, int]                    = kwargs.get("num_ports")       # number of chargers in each zone
    num_EVs         : int                               = kwargs.get("num_EVs")         # total number of EVs in the fleet
    charge_cost     : dict[int, float]                  = kwargs.get("charge_cost")     # price to charge one SoC level at time t
    
    # Metadata
    timestamp       : str                               = kwargs.get("timestamp", "")   # timestamp for logging
    file_name       : str                               = kwargs.get("file_name", "")   # filename for logging

    logger = Logger("model", level="DEBUG", to_console=True, timestamp=timestamp)

    logger.save("model_" + file_name) 
    logger.info("Parameters loaded successfully")

    # --------------------------
    # Sets
    # --------------------------
    ZONES       : list[int] = list (range(N))               # zones 0..N-1
    TIMESTEPS   : list[int] = list (range(T + 1))           # time steps 0..T
    LEVELS      : list[int] = list (range(L + 1))           # SoC levels 0..L
    AGES        : list[int] = list (range(W))               # ages 0..W-1, at age = W any unserved demand expires

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
    all_arcs            : dict[int                  , Arc]      = {}                                # Map all arcs to their unique ids
    type_arcs           : dict[ArcType              , set[int]] = {type: set() for type in ArcType} # sets of arc ids indexed by type
    in_arcs             : dict[Node                 , set[int]] = {v: set() for v in V}             # incoming arc ids per node
    out_arcs            : dict[Node                 , set[int]] = {v: set() for v in V}             # outgoing arc ids per node
    service_arcs_ijt    : dict[tuple[int, int, int] , set[int]] = {}                                # service arcs from node (i,t,l) to (j,t',l') for all l, t',l', indexed by (i,j,t)
    charge_arcs_it      : dict[tuple[int, int]      , set[int]] = {}                                # charging arcs from node (i,t,l) for all l, indexed by (i,t)

    # Associate revenue or cost with arcs by kind
    # Keys are arc ids
    arc_revenue     : dict[int, float] = {}     # service revenue
    arc_charge_cost : dict[int, float] = {}     # charging cost

    # Keep track the valid and invalid travel demand
    # We will only reference the valid ones
    valid_travel_demand    : dict[tuple[int, int, int], int]    = {}
    invalid_travel_demand  : set[tuple[int, int, int]]          = set()

    next_id = 0
    def _add_arc(
            type    : ArcType       , 
            o       : Node          ,     
            d       : Node          , 
        ) -> int | None:
        """
        Add an arc of a given type from origin node o to destination node d.
        Also registers the arc in all relevant sets.

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
        out_arcs[o].add(e.id)       # vertex is flowing from o -> d
        in_arcs[d].add(e.id)

        if type == ArcType.SERVICE:
            service_arcs_ijt.setdefault((o.i, d.i, o.t), set()).add(e.id)

        if type == ArcType.CHARGE:
            charge_arcs_it.setdefault((o.i, o.t), set()).add(e.id)

        return e.id
    
    def _add_service_arc (i, j, t):
        # We only add service arcs if there is demand
        if (i, j, t) not in travel_demand:
            return

        demand_ijt: int = travel_demand[(i, j, t)]

        if demand_ijt <= 0:
            invalid_travel_demand.add((i, j, t))
            logger.warning(f"No demand for service arc ({i}, {j}, {t}), skipping")
            return  # no demand, skip this arc

        travel_time_ijt : int = travel_time.get((i, j, t), 0)
        travel_energy_ij: int = travel_energy.get((i, j), 0)

        if travel_time_ijt <= 0 or travel_energy_ij <= 0:
            invalid_travel_demand.add((i, j, t))
            logger.warning(f"Invalid travel time or energy for service arc ({i}, {j}, {t}), skipping")
            return   # no valid travel time or energy, skip this arc

        if travel_time_ijt + t > T:
            invalid_travel_demand.add((i, j, t))
            logger.warning(f"Travel time exceeds total time T for service arc ({i}, {j}, {t}), skipping")
            return  # travel time exceeds total time T, skip this arc

        if travel_energy_ij > L:
            invalid_travel_demand.add((i, j, t))
            logger.warning(f"Travel energy exceeds max SoC L for service arc ({i}, {j}, {t}), skipping")
            return

        # if travel demand is made at t = t', we shall add service arc for t = t', t'+1, ..., t'+W-1
        # at t = t' + W, unserved demand leave the system and we incur penalty once for t = t' + W
        for wait_time in AGES:
            # check if service arc (i, j, t + wait_time) has already been added
            if (i, j, t + wait_time) in service_arcs_ijt:
                continue

            # check if t' + wait_time exceed T
            if t + wait_time > T:
                break

            travel_time_ijt_wait: int = travel_time.get((i, j, t + wait_time), 0)

            # check if travel_time from i to j at (t' + wait_time) is valid
            if travel_time_ijt_wait <= 0:
                continue
            
            # check if finishing the ride will exceed T
            if t + wait_time + travel_time_ijt_wait > T:
                continue
            
            if t + wait_time == 0:
                # EV start at max SoC
                o   : Node = Node(i, 0, L)
                d   : Node = Node(j, travel_time_ijt_wait, L - travel_energy_ij)
                e_id: int  = _add_arc(ArcType.SERVICE, o, d)

                arc_revenue[e_id] = order_revenue.get((i, j, 0), 0)
            else:
                for l in range(travel_energy_ij, L + 1):
                    # Need at least travel_energy_ij SoC to serve this demand                       
                    o   : Node = Node(i, t + wait_time, l)
                    d   : Node = Node(j, t + wait_time + travel_time_ijt_wait, l - travel_energy_ij)
                    e_id: int  = _add_arc(ArcType.SERVICE, o, d)

                    arc_revenue[e_id] = order_revenue.get((i, j, t + wait_time), 0)

            valid_travel_demand[(i, j, t)] = demand_ijt

    def _add_relocation_arc (i, j, t):
        # Only add relocation arcs if i != j and travel time is defined
        if (i == j) or ((i, j, t) not in travel_time):
            return

        travel_time_ijt = travel_time[(i, j, t)]
        travel_energy_ij = travel_energy.get((i, j), 0)

        if travel_time_ijt <= 0 or travel_energy_ij <= 0 or travel_time_ijt + t > T:
            return

        if t == 0:
            # EV start at max SoC
            o = Node(i, 0, L)
            d = Node(j, travel_time_ijt, L - travel_energy_ij)
            _add_arc(ArcType.RELOCATION, o, d)                        
        else:
            for l in range(travel_energy_ij, L + 1):
                # Need at least travel_energy_ij SoC to relocate
                o = Node(i, t, l)
                d = Node(j, t + travel_time_ijt, l - travel_energy_ij)
                _add_arc(ArcType.RELOCATION, o, d)
    
    def _add_charging_arc (i, j, t):
        if (t not in charge_cost) or (t + 1 > T) or (t == 0) or (i != j):
            return
        # Only add charging arcs if there is a charge cost defined for this time step
        # and no charging at first time step t = 0 (since EVs are fully charged)
        # EVs can only serve, relocate, or idle

        for l in LEVELS:
            # Charging arcs can only be added if there is enough room to charge

            o = Node(i, t, l)
            d = Node(i, t + 1, min (l + charge_speed, L))  # next level cannot exceed max SoC L

            # charging cost per EV in one time step = charging cost per level * charge speed (levels charged in one time step)
            cost = charge_cost.get(t) * charge_speed

            e_id = _add_arc(ArcType.CHARGE, o, d)
            arc_charge_cost[e_id] = cost

    def _add_idle_arc (i, j, t):
        # Idle arcs are added for every node at every time step
        if i != j:
            # idle when the EV never moved
            return

        if t == 0:
            o = Node(i, 0, L)
            d = Node(i, 1, L)
            _add_arc(ArcType.IDLE, o, d)                    
        elif t + 1 <= T:
            for l in LEVELS:
                o = Node(i, t, l)
                d = Node(i, t + 1, l)
                _add_arc(ArcType.IDLE, o, d)

    def _add_wraparound_arc (i, j, t):
        if (t != T) or (i != j):
            return
        
        for l in range (L_min, L + 1):
            # All EV must end the day with at least L_min SoC
            o = Node(i, T, l)
            d = Node(j, 0, L)  # since all EV start at max SoC, for flow conservation
            _add_arc(ArcType.WRAP, o, d)     

    # Add all arcs
    for i in ZONES:                 # Starting zone
        for j in ZONES:             # Destination zone
            for t in TIMESTEPS:     # Time when starting the trip
                _add_service_arc (i, j, t)
                _add_relocation_arc (i, j, t)
                _add_charging_arc (i, j, t)
                _add_idle_arc (i, j, t)
                _add_wraparound_arc (i, j, t)

    # Log any invalid demand that was skipped
    if invalid_travel_demand:
        logger.warning (f"Original number travel demand: {sum (travel_demand.values())} in {len(travel_demand)} entries")
        logger.warning (f"Valid number of travel demand: {sum (valid_travel_demand.values())} in {len(valid_travel_demand)} entries")
        logger.warning (f"Skipped {len(invalid_travel_demand)} invalid demand entries: {invalid_travel_demand}")
    
    logger.info(f"Arcs built with {len(all_arcs)} arcs")
    # Log the number of arcs by type
    for arc_type, arcs in type_arcs.items() :
        logger.info(f"  {arc_type.name} arcs: {len(arcs)}")
    
    # ----------------------------
    # Arcs Information
    # ----------------------------    
    logger.debug("Arcs information:")
    for id, arc in all_arcs.items():
        logger.debug (f"  Arc {id}: type {arc.type} from node ({arc.o.i}, {arc.o.t}, {arc.o.l}) to ({arc.d.i}, {arc.d.t}, {arc.d.l})")


    # ----------------------------
    # Model
    # ----------------------------
    with gp.Env() as env, gp.Model(env=env) as model:        
        model.Params.LogFile = f"Logs/gurobi_logs_{file_name}_{timestamp}.log"  # Set log file for Gurobi

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

        # s_ijt: number of cumulative unserved demand from zone i to j, considered at time interval t 
        #        = sum of unserved demand from t to t - W + 1
        #        (eg if t=3, w=2, then s consideres unserved demand for t=2 and t=3, whereas demand for t=1 has expired)
        s: gp.tupledict[tuple[int, int, int], gp.Var] = model.addVars(
            ((i, j, t) for i in ZONES for j in ZONES for t in TIMESTEPS) ,
            vtype   = GRB.INTEGER   , 
            lb      = 0             , 
            name    = "s"           ,
        )

        # u_ijta: number of unserved demand from zone i to j, considered at time interval t, of age a
        # eg a = 0: unserved demand from the demand made at t
        #    a = 1: unserved demand from the demand made at t-1
        u: gp.tupledict[tuple[int, int, int, int], gp.Var] = model.addVars(
            ((i, j, t, a) for i in ZONES for j in ZONES for t in TIMESTEPS for a in AGES) ,
            vtype   = GRB.INTEGER   , 
            lb      = 0             , 
            name    = "u"           ,
        )

        # e_ijt: number of unserved demand from zone i to j, that expired at t (can no longer be served)
        # ie if an demand expired at t, it must have been made at t - W
        e: gp.tupledict[tuple[int, int, int], gp.Var] = model.addVars(
            ((i, j, t) for i in ZONES for j in ZONES for t in TIMESTEPS) ,
            vtype   = GRB.INTEGER   , 
            lb      = 0             , 
            name    = "e"           ,
        )

        logger.info("Decision variables created")
        logger.info(f"  x_e variables: {len(x)}")
        logger.info(f"  s_ijt variables: {len(s)}")
        logger.info(f"  u_ijta variables: {len(u)}")
        logger.info(f"  e_ijt variables: {len(e)}")

        # ----------------------------
        # Constraints
        # ----------------------------

        # (1) Flow conservation at every node v ∈ V
        model.addConstrs(
            gp.quicksum(x[e_id] for e_id in out_arcs[v]) - gp.quicksum(x[e_id] for e_id in in_arcs[v]) == 0
            for v in V_set
        )
        logger.info("Flow conservation constraints (1) added")


        # (2) Accumulative unserved demand propagation
        #     unserved demand considered at t = unserved demand considered at t-1 + new demand at t - served demand at t - expired demand at t
        model.addConstrs(
            s[(i, j, t)] ==  s[(i, j, t-1)] + \
                valid_travel_demand.get((i, j, t), 0) - \
                gp.quicksum(
                    x[e_id] for e_id in service_arcs_ijt.get((i, j, t), set())
                ) - \
                e[(i, j, t)]
            for i in ZONES
            for j in ZONES
            for t in TIMESTEPS if t > 0
        )
        logger.info("Accumulative unserved demand propagation constraints (2) added")


        # (3) Accumulative unserved demand at t=0
        model.addConstrs(
            s[(i, j, 0)] \
                == valid_travel_demand.get((i, j, 0), 0) 
                    - gp.quicksum(
                        x[e] for e in service_arcs_ijt.get((i, j, 0), set())
                    )
                    - e[(i, j, 0)]
            for i in ZONES
            for j in ZONES
        )
        logger.info("Accumulative unserved demand at t=0 constraints (3) added")


        # (4) Accumulative unserved demand calculation
        #     unserved demand considered at t = unserved demand of age 0 + ... + unserved demand of age W-1
        model.addConstrs (
            s[(i, j, t)] == gp.quicksum (u[(i, j, t, a)] for a in AGES) 
            for i in ZONES
            for j in ZONES
            for t in TIMESTEPS
        )
        logger.info ("Accumulative unserved demand calculation constraints (4) added")


        # (5) Unserved demand aging for a > 0
        #     unserved demand with age a at time t propogates to unserved demand with age a+1 at time t+1
        model.addConstrs (
            u[(i, j, t-1, a-1)] >= u[(i, j, t, a)]
            for i in ZONES
            for j in ZONES
            for t in TIMESTEPS if t > 0
            for a in AGES if a > 0
        )
        logger.info ("Unserved demand aging for a > 0 constraints (5) added")


        # (6) Unserved demand aging base case
        #     unserved demand with age 0 at time t <= new demand at t
        model.addConstrs (
            valid_travel_demand.get((i, j, t), 0) >= u[(i, j, t, 0)]
            for i in ZONES
            for j in ZONES
            for t in TIMESTEPS
        )
        logger.info ("Unserved demand aging for a = 0 constraints (6) added")


        # (7) Unserved demand expiry for t < T
        #     unserved demand with age W-1 at time t becomes expired at t+1
        model.addConstrs (
            u[(i, j, t-1, W-1)] == e[(i, j, t)]
            for i in ZONES
            for j in ZONES
            for t in TIMESTEPS if t > 0
        )
        logger.info ("Unserved demand expiry for t < T constraints (7) added")


        # (9) Unserved demand base case
        #     unserved demand with age a at time t does not exist if t < a
        model.addConstrs (
            u[i, j, t, a] == 0
            for i in ZONES
            for j in ZONES
            for t in TIMESTEPS
            for a in AGES 
            if t < a
        )
        logger.info ("Unserved demand base case constraints (9) added")


        # (10) Expired demand base case
        #      expired demand does not exist for t < W
        model.addConstrs (
            e[i, j, t] == 0
            for i in ZONES
            for j in ZONES
            for t in TIMESTEPS
            if t < W
        )
        logger.info ("Expired demand base case constraints (10) added")


        # (11) Number of EVs charging at each zone at each time period cannot exceed the number of ports available in that zone
        model.addConstrs(
            gp.quicksum(x[e] for e in charge_arcs_it.get((i, t), set())) <= num_ports.get(i, 0)
            for i in ZONES  
            for t in TIMESTEPS
        )
        logger.info("Charging port constraints (11) added")


        # (12) Fleet size equals sum of wrap-around flows
        model.addConstr(
            gp.quicksum(x[e] for e in type_arcs[ArcType.WRAP]) == num_EVs
        )
        logger.info("Fleet size constraint (12) added")


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
                e[(i, j, t)] * penalty.get((i, j, t), 0)
                for i in ZONES
                for j in ZONES
                for t in TIMESTEPS
            ) \
            + gp.quicksum(
                s[(i, j, T)] * penalty.get((i, j, T), 0) # any unserved demand at the last time step T will become expired
                for i in ZONES 
                for j in ZONES
            ) ,
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
        model.write (f"Logs/gurobi_model_{file_name}_{timestamp}.lp")

        model.Params.DualReductions = 0 # debug infeasible or unbounded

        model.optimize()

        logger.info(f"Optimization completed with status {model.Status}")

        if model.Status != GRB.OPTIMAL:
            logger.error(f"Optimization was not successful. Status: {model.Status}")

            if model.Status == GRB.INFEASIBLE:
                model.computeIIS()
                model.write (f"Logs/gurobi_model_{file_name}_{timestamp}.ilp")
            return None
        
        logger.info("Optimization successful.")
        logger.info(f"  Runtime: {model.Runtime:.4f} seconds")
        logger.info(f"  Objective value: {model.ObjVal:.4f}")

        x_sol: dict[int, int]                       = {e: v.X for e, v in x.items()} # Keys are arc ids
        s_sol: dict[tuple[int, int, int], int]      = {k: v.X for k, v in s.items()} # Keys are (i, j, t) tuples
        u_sol: dict[tuple[int, int, int, int], int] = {k: v.X for k, v in u.items()}
        e_sol: dict[tuple[int, int, int], int]      = {k: v.X for k, v in e.items()}
        total_service_revenue_sol   : float     = total_service_revenue.X
        total_penalty_cost_sol      : float     = total_penalty_cost.X
        total_charge_cost_sol       : float     = total_charge_cost.X

        return {
            "obj"                       : model.ObjVal,
            "sol": {
                "x"                     : x_sol,
                "s"                     : s_sol,
                "u"                     : u_sol,
                "e"                     : e_sol,
                "total_service_revenue" : total_service_revenue_sol,
                "total_penalty_cost"    : total_penalty_cost_sol,
                "total_charge_cost"     : total_charge_cost_sol
            },
            "arcs": {
                "all_arcs"              : all_arcs,
                "type_arcs"             : type_arcs,
                "in_arcs"               : in_arcs,
                "out_arcs"              : out_arcs,
                "service_arcs_ijt"      : service_arcs_ijt,
                "charge_arcs_it"        : charge_arcs_it,
            },
            "sets": {
                "valid_travel_demand"  : valid_travel_demand,
                "invalid_travel_demand": invalid_travel_demand,
                "ZONES"                : ZONES,
                "TIMESTEPS"            : TIMESTEPS,
                "LEVELS"               : LEVELS,
                "AGES"                 : AGES,
            },
        }
    
        


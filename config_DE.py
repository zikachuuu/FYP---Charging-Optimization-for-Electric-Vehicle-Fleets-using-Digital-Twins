POP_SIZE        : int   = 16        # population size for DE (should be in multiples of <number of CPU cores> - 1)
NUM_CORES       : int   = 8         # number of CPU cores used
NUM_THREADS     : int   = 1         # number of threads used per candidate evaluation
                                    #   NUM_CORES * NUM_THREADS should be <= total available CPU cores
MAX_ITER        : int   = 2         # maximum iterations for DE
DIFF_WEIGHT     : float = 0.7       # Differential Weight / Mutation: Controls jump size
                                    #   Higher = bigger jumps (exploration); Lower = fine-tuning (exploitation)
                                    #   Since population size is small, need to aggresively explore to avoid getting stuck.
CROSS_PROB      : float = 0.7       # Crossover Probability: How much DNA comes from the mutant vs. the parent
                                    #   0.7 means 70% of the genes change every step.
                                    #   We want to mix good genes quickly, so set it high.
VAR_THRESHOLD   : float = 0         # variance threshold for early stopping of DE
PENALTY_WEIGHT  : float = 0.5       # penalty weight for high price settings in leader fitness function
NUM_ANCHORS     : int   = 6         # number of anchors for DE
VARS_PER_STEP   : int   = 3         # number of dimensions (variables) per time step (i.e. a_t, b_t, r_t)

RELAX_STAGE_2   : bool  = True      # whether to relax integrality constraints in follower model (Stage 2) during DE evaluations (for debugging)

"""
Notes for setting POP_SIZE and MAX_ITER:
    - Time to iterate the entire population once = time to evaluate one candidate * ceil (POP_SIZE / NUM_CORES)
    - MAX_ITER = (desired max runtime) / (time to iterate entire population once)
    - Check time to evaluate one candidate by running only the follower model (Stage 2) by choosing model_choice = '1' in the main.py
"""

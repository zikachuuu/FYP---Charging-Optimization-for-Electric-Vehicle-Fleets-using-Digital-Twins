# FYP---Charging-Optimization-for-Electric-Vehicle-Fleets-using-Digital-Twins

## Introduction
This is the repository for my FYP project titled: 

> Grid-Aware Charging for Shared Autonomous Electric Vehicles: A Bi-Level Optimization Approach for Fleet Operations and Grid Resilience

(Original Title: Charging Optimization for Electric Vehicle Fleets using Digital Twins)

In this repository, you will find

- Standalone, fully functional source code of the model presented in the paper.

- Testcases used in the experiments presented in the paper.

You will NOT find

- Raw Results (e.g. csv, Excel workbook, figures) of the experiments presented in the paper, due to size constraints. You can retrieve them by running the code locally on your computer with the same testcases and parameters settings.

- Parameter files corresponding to each experiment presented in the paper. There is only a single, unified parameter file, `config_DE.py`. You should adjust the parameters before conducting each experiment. For your reference, the parameters set will be shown in console as well as the log file.

- Configuration file used to request the computational resources from supercomputer servers. We used ASPIRE 2A hosted by the National Supercomputing Centre (NSCC) Singapore, and requested for 96 cores and 394 GB of memory. The configuration file specific to ASPIRE 2A, so you should write your own depending on the supercomputer server that you are using. Also, you should adjust the population size related parameters in `config_DE.py` depending on the amount of computational resources you have.

## To Start

1. Clone the repository

2. Run `pip install -r requirements.txt` <br>
    (Recommended: Create and activate a virtual environment first)

3. Create a `.env` file by following the format in `.env.sample` <br>
    (You need to have Gurobi installed and have a valid license)

4. Create a `config_DE.py` file by following the format in `config_DE.sample.py` <br> 

    The parameters have been categorized into 4 types:

    - Population size related parameters: It controls how long the model will run, more specifically, stage 1 (Differential Evolution). 

        - Adjust `POP_SIZE`, `NUM_PROCESSES`, `NUM_THREADS` based on number of CPU cores and size of memory avaliable. It does not have any effect on the results.

        - `MAX_ITER` and `VAR_THRESHOLD` has an effect on the results. Terminating earlier may lead to worse results. For optimal results, try to let the model for as many iterations as possible.

    - DR Randomness parameters: It controls how aggressive the search space is explored in each iteration. If the fitness or variance improvement after each iteration is minimal (e.g. high dimensional search space / stuck in local optimal), adjusting these parameters might help.

        - Adjust `PENALTY_WEIGHT` to compare the model that prioritize the objective of one stakeholder over another.

    - Bounds for decision variables: They are never changed throughout the experiments and there shouldn't be a need to.

    - Miscellaneous parameters: Only `RELAX_STAGE_2` will have to be adjusted. If the original integer programming model takes very long to solve and you just need some quick results (e.g. for analysis & debugging), you can set it to True to have a relaxed linear programming to be solved (usually takes less than a minute).
    
5. (Optional) Build your own testcase (json format only) and place it in the `Testcases` folder.

6. Run `main.py` and follow any instructions given. You will be asked to choose to run only Stage 1 (Bilevel Optimization to find optimal price), only Stage 2 (Follower Model), or run both stages.

    - If you choose to run only Stage 1, the optimal retail electricity pricing structure (base price, additional price, threshold) will be appended to the provided testcase (as `charge_cost_low`, `charge_cost_high`, `elec_threshold`) and return as a new json file in `Results` folder. This allows you to run both stages in two different sittings / on two different computers.

    - If you choose to run only Stage 2, then the  retail electricity pricing structure (as `charge_cost_low`, `charge_cost_high`, `elec_threshold`) should be in the provided testcase.

7. View results in `Results` folder and logs in `Logs` folder







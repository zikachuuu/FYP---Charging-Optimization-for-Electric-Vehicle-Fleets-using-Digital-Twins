# FYP---Charging-Optimization-for-Electric-Vehicle-Fleets-using-Digital-Twins

## What is this?


## Steps to run
1. Clone the repository
2. `pip install requirements.txt`
3. Create `.env` by following the format in `.env.sample`
4. (Optional) Build your own testcase (json format only) by following the format in `test1.json`, and place it in `Testcases` folder.
5. Run `main.py` and follow any instructions given
6. View results in `Results` folder and logs in `Logs` folder

## QnA
1. What does each parameter in the input testcase file stands for? <br>
Ans: Check their meaning in `Testcases/json_explain.txt`.

2. Where does each EV start at? <br>
Ans: There is no restriction on the number of EVs placed at each zone at the beginning. The model will choose the optimal number of EV to start at each zone.

3. What is the starting SoC of each EV? <br>
Ans: All EV start with full SoC `L` and must end the day equal to or above some certain threshold specified `L_min`.

4. How is revenue rewarded for each ride fulfilled? <br>
Ans: Revenue is rewarded based on the time it was actually fulfilled, not the time the ride request was made. For example, if a ride request was made at `t = 2` with $10 reward, and served at `t = 4` with $20 reward, the revenue rewarded will be $20.

5. How is the penalty for unserved demand calculated? <br>
Ans: An unserved ride made at `t` will stay in the system for up to `max_wait_time` number of time intervals. At `t + max_wait_time`, the unserved ride expires (leave the system) and penalty will be incurred based on the time it expires, not the time the ride request was made. At the last time interval `T`, all unserved demand become expired and penalty will be incurred for `T`.

6. My EV is not serving all demands! <br>
Ans: Some possible things to check:
    - Is your `order_revenue` too low compared to `charge_cost` or `travel_energy`?
    - Try setting `penalty` to a higher value.
    - All EVs must end the day with at least `L_min` SoC. Try lowering `L_min`, increasing `L`, increasing `num_EVs`, increasing `charge_speed`, or increasing `num_ports` at each zone if you see EVs spends most of their time charging.

7. The model status after optimization is not optimal (e.g. infeasible, unbounded)<br>
Ans: This should not occur and is a bug. Inform me together with all the logs files.






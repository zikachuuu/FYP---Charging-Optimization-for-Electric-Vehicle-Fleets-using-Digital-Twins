# FYP---Charging-Optimization-for-Electric-Vehicle-Fleets-using-Digital-Twins

## Steps to run
1. Clone the repository
2. `pip install requirements.txt`
3. Create `.env` by following the format in `.env.sample`
4. (Optional) Build your own testcase (json format only) by following the format in `test1.json`, and place it in `Testcases` folder.
5. Run `main.py` and follow any instructions given
6. View results and logs in `Logs` folder

## QnA
1. My EV is not serving all demand no matter how much I increase the EV fleet size! <br>
Ans: Check your number of charging ports. EVs must end at the SoC level they begin with (or higher). If an EV is not serving a ride, it is likely because there is not enough ports for the EV to charge after the ride end.

2. What is the starting SoC of each EV? <br>
Ans: This is purely decided by the model. Also there is no penalty for assigning higher starting SoC to EVs. (A bit unreasonable if you ask me)

3. How is penalty for unserved demand calculated? <br>
Ans: A unserved ride will incur penaty for each time period it stays in the system. For example, if a ride was made at t2 and only served at t5, then penalty will be incurred for t2, t3, and t4. This is to incentivise model to fulfill all ride requests as early as possible.

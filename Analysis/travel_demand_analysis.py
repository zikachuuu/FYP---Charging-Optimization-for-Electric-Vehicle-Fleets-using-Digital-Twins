import json
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "figure.titlesize": 18,
})


# Load test case
base_dir = os.path.dirname(os.path.abspath(__file__))
testcase_path = os.path.normpath(os.path.join(base_dir, "..", "Testcases", "manhattan_jan2.json"))
with open(testcase_path, 'r') as f:
    testcase = json.load(f)

# Extract parameters
T = testcase["T"]  # Number of time steps
N = testcase["N"]  # Number of zones
travel_demand = testcase["travel_demand"]

# Aggregate travel demand by time step
demand_by_time = [0] * (T + 1)

for key, demand_value in travel_demand.items():
    parts = key.split(',')
    origin = int(parts[0])
    destination = int(parts[1])
    time = int(parts[2])
    
    if 0 <= time <= T:
        demand_by_time[time] += demand_value

# Calculate total demand
total_demand = sum(demand_by_time)

print(f"\n{'='*60}")
print(f"Travel Demand Analysis for {testcase_path}")
print(f"{'='*60}")
print(f"Number of zones (N): {N}")
print(f"Number of time steps (T): {T}")
print(f"\nTotal travel demand across all time steps: {total_demand} rides")
print(f"Average demand per time step: {total_demand / (T + 1):.2f} rides")
print(f"Peak demand at time step: {demand_by_time.index(max(demand_by_time))}")
print(f"Peak demand value: {max(demand_by_time)} rides")
print(f"{'='*60}\n")

# Create plot
plt.figure(figsize=(12, 8))
time_steps = list(range(T + 1))

plt.plot(
    time_steps,
    demand_by_time,
    color='tab:blue',
    linewidth=2.5,
    marker='o',
    markersize=6,
    label='Total Travel Demand'
)

plt.fill_between(
    time_steps,
    demand_by_time,
    alpha=0.3,
    color='tab:blue'
)

plt.xlabel('Time Step')
plt.ylabel('Travel Demand (number of rides)')
plt.title(f'Travel Demand Throughout Time Steps')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1, frameon=True)
plt.tight_layout()

# Save plot
output_path = "travel_demand_analysis.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {output_path}\n")

plt.show()

# Print detailed time step breakdown
print("\nDetailed Travel Demand by Time Step:")
print(f"{'Time Step':<12} {'Demand':<12}")
print("-" * 24)
for t in time_steps:
    print(f"{t:<12} {demand_by_time[t]:<12}")
print("-" * 24)
print(f"{'TOTAL':<12} {total_demand:<12}")

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
order_revenue = testcase["order_revenue"]

# Aggregate by time step
demand_by_time = [0] * (T + 1)
revenue_by_time = [0.0] * (T + 1)

for key, demand_value in travel_demand.items():
    parts = key.split(',')
    origin = int(parts[0])
    destination = int(parts[1])
    time = int(parts[2])
    
    if 0 <= time <= T:
        demand_by_time[time] += demand_value

for key, revenue_value in order_revenue.items():
    parts = key.split(',')
    origin = int(parts[0])
    destination = int(parts[1])
    time = int(parts[2])
    
    if 0 <= time <= T:
        revenue_by_time[time] += revenue_value

# Calculate average revenue per ride
avg_revenue_per_ride = []
for t in range(T + 1):
    if demand_by_time[t] > 0:
        avg_revenue_per_ride.append(revenue_by_time[t] / demand_by_time[t])
    else:
        avg_revenue_per_ride.append(0)

# Calculate totals
total_demand = sum(demand_by_time)
total_revenue = sum(revenue_by_time)
overall_avg_revenue = total_revenue / total_demand if total_demand > 0 else 0

# Find peak average revenue
peak_avg_idx = 0
peak_avg_value = 0
for t in range(T + 1):
    if avg_revenue_per_ride[t] > peak_avg_value:
        peak_avg_value = avg_revenue_per_ride[t]
        peak_avg_idx = t

print(f"\n{'='*70}")
print(f"Order Revenue Analysis for {testcase_path}")
print(f"{'='*70}")
print(f"Total travel demand: {total_demand} rides")
print(f"Total revenue: ${total_revenue:,.2f}")
print(f"Overall average revenue per ride: ${overall_avg_revenue:.2f}")
print(f"\nPeak average revenue: ${peak_avg_value:.2f} at time step {peak_avg_idx}")
print(f"Lowest average revenue: ${min(avg_revenue_per_ride):.2f}")
print(f"{'='*70}\n")

# Create plot with dual axes
fig, ax1 = plt.subplots(figsize=(12, 8))

time_steps = list(range(T + 1))

# Left axis - Average revenue per ride
color = 'tab:green'
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Average Revenue per Ride ($)', color=color)
line1 = ax1.plot(
    time_steps,
    avg_revenue_per_ride,
    color=color,
    linewidth=2,
    marker='o',
    markersize=6,
    label='Average Revenue per Ride'
)
ax1.fill_between(
    time_steps,
    avg_revenue_per_ride,
    alpha=0.2,
    color=color
)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

# Right axis - Demand
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Travel Demand (rides)', color=color)
line2 = ax2.plot(
    time_steps,
    demand_by_time,
    color=color,
    linewidth=2,
    marker='s',
    markersize=5,
    label='Travel Demand',
    linestyle='--',
    alpha=0.7
)
ax2.tick_params(axis='y', labelcolor=color)

# Combined title and legend
plt.title(
    f'Average Revenue per Ride Throughout the Day',
    pad=20
)

# Combine legends from both axes
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=True)

plt.tight_layout()

# Save plot
output_path = "order_revenue_analysis.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {output_path}\n")

plt.show()

# Print detailed time step breakdown
print("\nDetailed Revenue Analysis by Time Step:")
print(f"{'Time':<6} {'Demand':<10} {'Total Revenue':<16} {'Avg Rev/Ride':<14}")
print("-" * 50)
for t in time_steps:
    demand = demand_by_time[t]
    revenue = revenue_by_time[t]
    avg_rev = avg_revenue_per_ride[t]
    print(f"{t:<6} {demand:<10} ${revenue:<15.2f} ${avg_rev:<13.2f}")
print("-" * 50)
print(f"{'TOTAL':<6} {total_demand:<10} ${total_revenue:<15.2f} ${overall_avg_revenue:<13.2f}")

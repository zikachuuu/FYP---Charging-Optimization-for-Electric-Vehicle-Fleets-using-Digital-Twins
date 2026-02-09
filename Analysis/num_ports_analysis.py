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
N = testcase["N"]  # Number of zones
num_ports = testcase["num_ports"]

# Convert to lists for plotting
zones = []
ports = []

for zone_str, port_count in num_ports.items():
    zone = int(zone_str)
    zones.append(zone)
    ports.append(port_count)

# Sort by zone
sorted_data = sorted(zip(zones, ports))
zones = [z for z, p in sorted_data]
ports = [p for z, p in sorted_data]

# Calculate statistics
total_ports = sum(ports)
avg_ports = total_ports / N
max_ports = max(ports)
min_ports = min(ports)
zones_with_no_ports = sum(1 for p in ports if p == 0)
zones_with_ports = N - zones_with_no_ports

print(f"\n{'='*70}")
print(f"Charging Ports Analysis for {testcase_path}")
print(f"{'='*70}")
print(f"Number of zones: {N}")
print(f"Total charging ports: {total_ports}")
print(f"Average ports per zone: {avg_ports:.2f}")
print(f"Maximum ports in a zone: {max_ports} (Zone {zones[ports.index(max_ports)]})")
print(f"Minimum ports in a zone: {min_ports}")
print(f"Zones with charging ports: {zones_with_ports}")
print(f"Zones without charging ports: {zones_with_no_ports}")
print(f"{'='*70}\n")

# Create bar plot
plt.figure(figsize=(14, 8))

# Color zones based on number of ports
colors = []
for p in ports:
    if p == 0:
        colors.append('lightgray')
    elif p < 5:
        colors.append('lightyellow')
    elif p < 10:
        colors.append('lightgreen')
    elif p < 15:
        colors.append('skyblue')
    else:
        colors.append('salmon')

bars = plt.bar(zones, ports, color=colors, edgecolor='black', linewidth=0.5)

# Add horizontal line for average
plt.axhline(y=avg_ports, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_ports:.2f} ports')

plt.xlabel('Zone ID')
plt.ylabel('Number of Charging Ports')
plt.title(f'Charging Ports Distribution Across Zones', 
          pad=20)
plt.grid(True, alpha=0.3, axis='y')
plt.legend(fontsize=12)

# Add value labels on top of bars (only for non-zero values to avoid clutter)
for i, (zone, port) in enumerate(zip(zones, ports)):
    if port > 0:
        plt.text(zone, port + 0.3, str(port), ha='center', va='bottom', fontsize=8)

plt.tight_layout()

# Save plot
output_path = "num_ports_analysis.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {output_path}\n")

plt.show()

# Print detailed zone breakdown
print("\nDetailed Charging Ports by Zone:")
print(f"{'Zone':<8} {'Ports':<10}")
print("-" * 20)
for zone, port in zip(zones, ports):
    print(f"{zone:<8} {port:<10}")
print("-" * 20)
print(f"{'TOTAL':<8} {total_ports:<10}")
print(f"{'AVG':<8} {avg_ports:<10.2f}")

# Distribution summary
print(f"\n{'='*70}")
print("Distribution Summary:")
print(f"{'='*70}")
print(f"Zones with 0 ports: {zones_with_no_ports} zones")
print(f"Zones with 1-4 ports: {sum(1 for p in ports if 1 <= p < 5)} zones")
print(f"Zones with 5-9 ports: {sum(1 for p in ports if 5 <= p < 10)} zones")
print(f"Zones with 10-14 ports: {sum(1 for p in ports if 10 <= p < 15)} zones")
print(f"Zones with 15+ ports: {sum(1 for p in ports if p >= 15)} zones")
print(f"{'='*70}")

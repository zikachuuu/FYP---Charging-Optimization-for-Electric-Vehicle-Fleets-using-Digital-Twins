import json
from collections import OrderedDict

input_file = "Testcases/scenarios/manhattan_jan2_ports.json"
print(f"Reading {input_file}...")
with open(input_file, 'r') as f:
    data = json.load(f)

# Convert num_ports to int
num_ports = data.get('num_ports', {})
updated_num_ports = OrderedDict()
for zone, value in num_ports.items():
    updated_num_ports[zone] = int(value)

# Convert elec_supplied to int
elec_supplied = data.get('elec_supplied', {})
updated_elec_supplied = OrderedDict()
for key, value in elec_supplied.items():
    updated_elec_supplied[key] = int(value)

print(f"Converted {len(updated_num_ports)} num_ports entries to int")
print(f"Converted {len(updated_elec_supplied)} elec_supplied entries to int")

# Replace in data
data['num_ports'] = updated_num_ports
data['elec_supplied'] = updated_elec_supplied

# Write back
with open(input_file, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Done! All values converted to integers")

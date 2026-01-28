def convert_key_types(data: dict) -> dict:
    """
    Convert string keys to appropriate types (tuples or integers) based on their format.
    - Keys with commas (e.g., "0,1,2") become tuples of integers
    - Keys that are pure integers (e.g., "5") become integers
    - Other keys remain as strings
    """
    def parse_key(k):
        """Parse a single key to its appropriate type."""
        if not isinstance(k, str):
            return k  # Return non-string keys as-is
        
        # Check if it contains commas (tuple format)
        if ',' in k:
            try:
                parts = k.split(',')
                return tuple(int(p.strip()) for p in parts)
            except ValueError:
                return k  # Keep as string if conversion fails
        
        # Try to convert to integer
        try:
            return int(k.strip())
        except ValueError:
            return k  # Keep as string if not an integer
    
    def convert_dict_keys(d):
        """Recursively convert keys in a dictionary."""
        if not isinstance(d, dict):
            return d
        
        new_dict = {}
        for k, v in d.items():
            new_key = parse_key(k)
            # Recursively process nested dictionaries
            if isinstance(v, dict):
                new_dict[new_key] = convert_dict_keys(v)
            else:
                new_dict[new_key] = v
        return new_dict
    
    # Process the entire data structure
    return convert_dict_keys(data)

def print_duration (time_in_seconds: float) -> str:
    """Convert time duration in seconds to a human-readable string format."""
    minutes, seconds = divmod(time_in_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    
    duration_parts = []
    if hours > 0:
        duration_parts.append(f"{int(hours)} hours")
    if minutes > 0:
        duration_parts.append(f"{int(minutes)} minutes")
    duration_parts.append(f"{seconds:.2f} seconds")
    
    return ', '.join(duration_parts)
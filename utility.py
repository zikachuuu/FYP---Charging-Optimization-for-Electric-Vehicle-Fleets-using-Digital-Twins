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


def round_df(df):
    """
    Round numeric values in a DataFrame to 3 decimal places.
    Returns a new DataFrame with numeric columns rounded.
    """
    try:
        return df.round(3)
    except Exception:
        # Fallback: round numeric columns in-place
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            df[col] = df[col].round(3)
        return df

def weighted_quantile(values, weights, quantile):
    """
    Compute weighted quantile for discrete values.

    Args:
        values: list of numeric values
        weights: list of non-negative weights
        quantile: float in [0, 1]

    Returns:
        float: weighted quantile value
    """
    if not values or not weights:
        return float("nan")

    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total_weight = sum(w for _, w in pairs)
    if total_weight <= 0:
        return float("nan")

    target = quantile * total_weight
    cumulative = 0.0
    for v, w in pairs:
        cumulative += w
        if cumulative >= target:
            return float(v)
    return float(pairs[-1][0])
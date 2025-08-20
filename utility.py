def convert_key_types(data: dict) -> dict:
    def to_tuple_keyed(d: dict, tuple_len: int) -> dict:
        new_d = {}
        for k, v in d.items():
            if isinstance(k, str):
                parts = k.split(",")
                if len(parts) != tuple_len:
                    raise ValueError(f"Key {k!r} does not have {tuple_len} parts")
                try:
                    tkey = tuple(int(p) for p in parts)
                except ValueError as e:
                    raise ValueError(f"Key {k!r} has non-integer component") from e
            elif isinstance(k, tuple) and len(k) == tuple_len:
                # already a tuple, ensure ints
                try:
                    tkey = tuple(int(p) for p in k)
                except ValueError as e:
                    raise ValueError(f"Key {k!r} has non-integer component") from e
            else:
                raise TypeError(f"Unexpected key type {type(k)} for key {k!r}")
            new_d[tkey] = v
        return new_d

    def to_int_keyed(d: dict) -> dict:
        new_d = {}
        for k, v in d.items():
            if isinstance(k, str):
                try:
                    ikey = int(k)
                except ValueError as e:
                    raise ValueError(f"Key {k!r} is not an int") from e
            elif isinstance(k, int):
                ikey = k
            else:
                raise TypeError(f"Unexpected key type {type(k)} for key {k!r}")
            new_d[ikey] = v
        return new_d

    out = dict(data)  # shallow copy

    # 3-tuple keyed dicts
    for name in ("travel_demand", "travel_time", "order_revenue", "penalty"):
        if name in out and isinstance(out[name], dict):
            out[name] = to_tuple_keyed(out[name], 3)

    # 2-tuple keyed dicts
    for name in ("travel_energy",):
        if name in out and isinstance(out[name], dict):
            out[name] = to_tuple_keyed(out[name], 2)

    # int-keyed dicts
    for name in ("num_ports", "charge_cost"):
        if name in out and isinstance(out[name], dict):
            out[name] = to_int_keyed(out[name])

    return out
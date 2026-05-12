# -- Flatten a Dictionary --
# function that turns a nested dictionary
# into an unnested dictionary with nested
# keys concatenated with each other
def flatten_dict(d, parent_key='', sep='_'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep))
        else:
            # note that we return each item in a list rather than the item itself,
            # this is because we want to flatten dicts for batch processing,
            # which requires a list or other collection
            items[new_key] = [v]
    return items

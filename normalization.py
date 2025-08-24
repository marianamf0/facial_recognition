def normalize_zscore(value, axis:int=1): 
    mean = value.mean(axis=axis, keepdims=True)
    std  = value.std(axis=axis, ddof=0, keepdims=True)
    std[std == 0.0] = 1.0
    return (value - mean) / std
    
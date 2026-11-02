import numpy as np
from collections import Counter

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Your code here
    data = np.asarray(data)

    if data.size == 0:
        raise ValueError("Dataset can't be empty")

    counts = Counter(data.tolist())
    max_count = max(counts.values())
    mode = min(k for k, v in counts.items() if v == max_count)

    mean = np.mean(data)
    median = np.median(data)
    variance = np.var(data)
    std = np.std(data)

    p25, p50, p75 = np.percentile(data, [25, 50, 75])

    return {
        "mean": float(mean),
        "median": float(median),
        "mode": float(mode),
        "variance": float(variance),
        "standard_deviation": float(std),
        "25th_percentile": float(p25),
        "50th_percentile": float(p50),
        "75th_percentile": float(p75),
        "interquartile_range": float(p75- p25)
    }
import numpy as np

def flag_outliers(array, iqr_distance = 1.5):
    upper_quartile = np.percentile(array, 75)
    lower_quartile = np.percentile(array, 75)
    IQR = (upper_quartile - lower_quartile) * iqr_distance
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    f = np.logical_and(array > quartileSet[0], array < quartileSet[1])

    return f

import numpy as np

def flag_outliers(array, iqr_distance = 1.5):
    
    upper_quartile = np.nanpercentile(array, 75)
    lower_quartile = np.nanpercentile(array, 25)
    IQR = (upper_quartile - lower_quartile) * iqr_distance
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    
    fiqr = np.logical_and(array > quartileSet[0], array < quartileSet[1])

    fpos = array > 0

    f = np.logical_and(fiqr, fpos)

    print('{} samples was flagged as outliers.'.format(np.sum(~f)))

    return f
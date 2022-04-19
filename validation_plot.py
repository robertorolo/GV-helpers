import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def validation_plot(estimated, true, title, outfl):

    edefined = np.isfinite(estimated)
    tdefined = np.isfinite(true)
    bothdefined = np.logical_and(edefined, tdefined)
    estimated = estimated[bothdefined]
    true = true[bothdefined]

    fig, axs = plt.subplots(1, 3, figsize=(20,6))
    min_val = np.min([np.nanmin(estimated), np.nanmin(true)])
    max_val = np.max([np.nanmax(estimated), np.nanmax(true)])

    axs[0].plot([min_val, max_val], [min_val, max_val], color='red')
    slope, intercept, r_value, p_value, std_err = stats.linregress(estimated,true)
    x_d = np.linspace(min_val, max_val, 100)
    y_r = slope*x_d+intercept

    bias = true-estimated
    mse = np.mean(bias**2)

    statsvals = '''
    n {}
    MSE {}
    Slope {}
    Pearson rho {}
    '''.format(len(estimated), round(mse, 2), round(slope, 2), round(r_value, 2))

    axs[0].plot(x_d, y_r, color='gray', linestyle='--', label='Regression line')
    axs[0].annotate(statsvals, xy=(0.55, 0.0), xycoords='axes fraction', color='black')

    axs[0].scatter(estimated, true, c='black')
    axs[0].set_ylim([min_val, max_val])
    axs[0].set_xlim([min_val, max_val])
    axs[0].grid()
    axs[0].set_ylabel('True')
    axs[0].set_xlabel('Estimated')

    axs[1].hist(bias, bins=20, color='green')
    statsvals = '''
    Mean: {}
    Std: {}
    '''.format(round(np.mean(bias), 2), round(np.std(bias), 2))
    axs[1].annotate(statsvals, xy=(0.6, 0.85), xycoords='axes fraction', color='black')
    axs[1].grid()

    axs[2].scatter(true, bias, color='black')
    axs[2].grid()
    axs[2].set_ylabel('Error')
    axs[2].set_xlabel('Grade')
    axs[2].axhline(0, color='red')

    #saving
    plt.suptitle(title)
    plt.tight_layout()

    plt.savefig(outfl, facecolor='white', bbox_inches='tight')
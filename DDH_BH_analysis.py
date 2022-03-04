import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class DDHBHanalysis:
    def __init__(self, var_name, ddh_array, bh_array, KSAlpha=0.05):
        if ddh_array.size != bh_array.size:
            print('Array do not have the same size!')
        else:
            self.var_name = var_name
            self.ddh_array = ddh_array
            self.bh_array = bh_array
            self.KSAlpha = KSAlpha

    def plot_analysis(self, outfl=None):
        if outfl is None:
            outfl = "{}_ddhbhanalysis.png".format(self.var_name)
        
        fig, axs = plt.subplots(1, 3, figsize=(20,5))

        #defining minimum and maximum values
        min_val = np.min([np.nanmin(self.bh_array), np.nanmin(self.ddh_array)])
        max_val = np.max([np.nanmax(self.bh_array), np.nanmax(self.ddh_array)])

        #scatter plot
        axs[0].plot([min_val, max_val], [min_val, max_val], color='red')
        axs[1].plot([min_val, max_val], [min_val, max_val], color='red')

        axs[0].plot([min_val*(1+0.2), max_val*(1+0.2)], [min_val, max_val], color='blue', alpha=0.5, label='20% dispersion')
        axs[0].plot([min_val*(1-0.2), max_val*(1-0.2)], [min_val, max_val], color='blue', alpha=0.5)

        axs[1].plot([min_val*(1+0.2), max_val*(1+0.2)], [min_val, max_val], color='blue', alpha=0.5, label='20% dispersion')
        axs[1].plot([min_val*(1-0.2), max_val*(1-0.2)], [min_val, max_val], color='blue', alpha=0.5)

        slope, intercept, r_value, p_value, std_err = stats.linregress(self.bh_array,self.ddh_array)
        x_d = np.linspace(min_val, max_val, 100)
        y_r = slope*x_d+intercept
        mse = np.sqrt(((self.bh_array - self.ddh_array) ** 2).mean())
        rel_error = np.mean(np.absolute(self.ddh_array-self.bh_array)/self.ddh_array * 100)
        statsvals = '''
        n {}
        MSE {}
        MRE {}%
        Slope {}
        R-squared {}
        '''.format(len(self.bh_array), mse.round(2), rel_error.round(2), slope.round(2), r_value.round(2))
        axs[0].plot(x_d, y_r, color='gray', linestyle='--', label='Regression line')
        axs[0].annotate(statsvals, xy=(0.6, 0.0), xycoords='axes fraction')
        axs[0].scatter(self.bh_array, self.ddh_array, color='black')

        axs[0].set_title('Scatterplot')
        axs[0].set_ylabel('DDH')
        axs[0].set_xlabel('BH')
        axs[0].set_ylim([min_val, max_val])
        axs[0].set_xlim([min_val, max_val])
        axs[0].legend(loc='upper left')
        axs[0].grid()

        #qqplot
        x_quant = [np.nanquantile(np.array(self.bh_array), q) for q in np.linspace(0,1,self.bh_array.size)]
        y_quant = [np.nanquantile(np.array(self.ddh_array), q) for q in np.linspace(0,1,self.bh_array.size)]
        axs[1].scatter(x_quant, y_quant, color='black')

        axs[1].set_title('QQ-plot')
        axs[1].set_ylabel('DDH')
        axs[1].set_xlabel('BH')
        axs[1].set_ylim([min_val, max_val])
        axs[1].set_xlim([min_val, max_val])
        axs[1].legend(loc='upper left')
        axs[1].grid()

        #ks test
        statistic, pvalue = stats.ks_2samp(self.bh_array, self.ddh_array)
        c = np.sqrt(-np.log(self.KSAlpha/2)*2/(2*self.bh_array.size))
        if c < statistic:
            decision = 'Reject'
        else:
            decision = 'Accept'
        statsvals = '''
        D statistic {}
        p-value {}
        Treshold  {}
        {}
        '''.format(round(statistic,5), round(pvalue,5), round(c, 5), decision)
        axs[2].annotate(statsvals, xy=(0.6, 0.0), xycoords='axes fraction')
        bh_y = np.linspace(0, 1, len(self.bh_array), endpoint=True)
        lt_y = np.linspace(0, 1, len(self.ddh_array), endpoint=True)
        axs[2].plot(np.sort(self.bh_array), bh_y, label='BH')
        axs[2].plot(np.sort(self.ddh_array), lt_y, label='DDH')
        axs[2].set_title('KS test - eCDF')
        axs[2].set_ylim([0,1])
        axs[2].set_xlim([min_val,max_val])
        axs[2].legend(loc='upper right')
        axs[2].grid()

        #saving
        plt.suptitle('{} - DDH-BH analysis'.format(self.var_name))

        plt.savefig(outfl, dpi=300, facecolor='white', bbox_inches='tight')

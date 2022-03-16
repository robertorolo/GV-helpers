import numpy as np
from scipy import stats
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
from scipy.stats import chi2
import matplotlib.pyplot as plt

def flag_outlier(x, y):
    X = np.array([x, y]).T
    mdists = np.array([mahalanobis(i , np.mean(X,axis=0), inv(np.cov(X.T))) for i in X])
    p = 1 - chi2.cdf(mdists, 1)
    outlier_flag = p < .001

    return outlier_flag

class DDHBHanalysis:
    def __init__(self, var_name, ddh_array, bh_array, KSAlpha=0.05):
        if ddh_array.size != bh_array.size:
            print('Arrays do not have the same size!')

        else:
            
            ddhdefined = np.isfinite(ddh_array)
            bhdefined = np.isfinite(bh_array)
            bothdefined = np.logical_and(ddhdefined, bhdefined)
            self.var_name = var_name
            ddh_array = ddh_array[bothdefined]
            bh_array = bh_array[bothdefined]
            self.KSAlpha = KSAlpha

            outf = flag_outlier(ddh_array, bh_array)
            self.ddh_all = ddh_array
            self.bh_all = bh_array
            self.ddh_array = ddh_array[~outf]
            self.bh_array = bh_array[~outf]
            self.ddh_out = ddh_array[outf]
            self.bh_out = bh_array[outf]
            self.num_outf = np.sum(outf)

    def plot_analysis(self, outfl=None):
        if self.ddh_array.size == 0 or self.bh_array.size == 0:
            print('Empty array!')
        else:
            if outfl is None:
                outfl = "{}_ddhbhanalysis.png".format(self.var_name)
            
            fig, axs = plt.subplots(1, 3, figsize=(20,5))

            #defining minimum and maximum values
            min_val = np.min([np.nanmin(self.bh_all), np.nanmin(self.ddh_all)])
            max_val = np.max([np.nanmax(self.bh_all), np.nanmax(self.ddh_all)])

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
            bias = self.bh_array-self.ddh_array
            rel_bias = np.mean(bias)/np.std(bias)
            statsvals = '''
            n {}
            MSE {}
            Relative bias {}
            Slope {}
            R-squared {}
            removed outliers {}
            '''.format(len(self.bh_array), mse.round(2), rel_bias.round(2), slope.round(2), r_value.round(2), self.num_outf)
            axs[0].plot(x_d, y_r, color='gray', linestyle='--', label='Regression line')
            axs[0].annotate(statsvals, xy=(0.55, 0.0), xycoords='axes fraction', color='gray')
            axs[0].scatter(self.bh_array, self.ddh_array, color='black')
            if self.num_outf != 0:
                axs[0].scatter(self.bh_out, self.ddh_out, color='green', label='outliers')

            axs[0].set_title('Scatterplot')
            axs[0].set_ylabel('DDH: mean {} - std {}'.format(round(np.mean(self.ddh_array), 2), round(np.std(self.ddh_array), 2)))
            axs[0].set_xlabel('BH: mean {} - std {}'.format(round(np.mean(self.bh_array), 2), round(np.std(self.bh_array), 2)))
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
            axs[2].annotate(statsvals, xy=(0.55, 0.0), xycoords='axes fraction', color='gray')
            bh_y = np.linspace(0, 1, len(self.bh_array), endpoint=True)
            lt_y = np.linspace(0, 1, len(self.ddh_array), endpoint=True)
            axs[2].plot(np.sort(self.bh_array), bh_y, label='BH', color='blue')
            axs[2].plot(np.sort(self.ddh_array), lt_y, label='DDH', color='red')
            axs[2].set_title('KS test - eCDF')
            axs[2].set_ylim([0,1])
            axs[2].set_xlim([min_val,max_val])
            axs[2].legend(loc='upper right')
            axs[2].grid()

            #saving
            plt.suptitle('{} - DDH-BH analysis'.format(self.var_name))

            plt.savefig(outfl, dpi=300, facecolor='white', bbox_inches='tight')
import numpy as np
from scipy import stats
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
from scipy.stats import chi2
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.spatial import KDTree

def find_twins(dx, dy, dz, bx, by, bz, r):
    bh_coords = np.array([bx, by, bz]).T
    dh_coords = np.array([dx, dy, dz]).T
    tree = KDTree(bh_coords)

    dhs = []
    bh_twins = []
    picked = []

    for idx, p in enumerate(dh_coords):
        indices = tree.query_ball_point(p, r)
        if len(indices) != 0:
            for jdx, indice in enumerate(indices):
                if indice in picked:
                    pass
                else:
                    picked.append(indice)
                    dhs.append(idx)
                    bh_twins.append(indice)
                    continue

    return dhs, bh_twins

def interpolation(d, x):
    output = d[0][1] + (x - d[0][0]) * ((d[1][1] - d[0][1])/(d[1][0] - d[0][0]))
    
    return output

def flag_outlier(x, y):
    X = np.array([x, y]).T
    mdists = np.array([mahalanobis(i , np.mean(X,axis=0), inv(np.cov(X.T))) for i in X])
    p = 1 - chi2.cdf(mdists, 1)
    outlier_flag = p < .001

    return outlier_flag

class DDHBHanalysis:
    def __init__(self, var_name, ddh_array, bh_array, KSAlpha=0.05, logscale=False):
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
            self.logscale = logscale

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
            
            fig, axs = plt.subplots(1, 5, figsize=(35,6))

            #defining minimum and maximum values
            min_val = np.min([np.nanmin(self.bh_all), np.nanmin(self.ddh_all)])
            max_val = np.max([np.nanmax(self.bh_all), np.nanmax(self.ddh_all)])

            min_val_out = np.min([np.nanmin(self.bh_array), np.nanmin(self.ddh_array)])
            max_val_out = np.max([np.nanmax(self.bh_array), np.nanmax(self.ddh_array)])

            #scatter plot
            axs[0].plot([min_val, max_val], [min_val, max_val], color='red')
            axs[2].plot([min_val_out, max_val_out], [min_val_out, max_val_out], color='red')

            axs[0].plot([min_val*(1+0.2), max_val*(1+0.2)], [min_val, max_val], color='blue', alpha=0.5, label='20% dispersion')
            axs[0].plot([min_val*(1-0.2), max_val*(1-0.2)], [min_val, max_val], color='blue', alpha=0.5)

            axs[2].plot([min_val_out*(1+0.2), max_val_out*(1+0.2)], [min_val_out, max_val_out], color='blue', alpha=0.5, label='20% dispersion')
            axs[2].plot([min_val_out*(1-0.2), max_val_out*(1-0.2)], [min_val_out, max_val_out], color='blue', alpha=0.5)

            #line parameters
            slope, intercept, r_value, p_value, std_err = stats.linregress(self.bh_array,self.ddh_array)
            x_d = np.linspace(min_val, max_val, 100)
            y_r = slope*x_d+intercept

            #bias
            bias = self.ddh_array-self.bh_array
            #relative bias
            #rel_bias = 2*100*bias/(self.ddh_array+self.bh_array) #relative change
            rel_bias = 100 * bias/self.ddh_array
            #mean rel bias
            mean_rel_bias = np.mean(rel_bias)
            #abs_rel_bias
            abs_rel_bias = np.abs(rel_bias)
            #Maxbias
            max_bias = np.max(abs_rel_bias)
            statsvals = '''
            n {}
            Mean relative error {} %
            Max relative error {} %
            Slope {}
            Pearson rho {}
            Removed outliers {}
            '''.format(len(self.bh_array), round(mean_rel_bias, 2), round(max_bias, 2),  round(slope, 2), round(r_value, 2), self.num_outf)
            axs[0].plot(x_d, y_r, color='gray', linestyle='--', label='Regression line')
            axs[0].annotate(statsvals, xy=(0.34, 0.0), xycoords='axes fraction', color='black')
            
            #scatter kde
            xy = np.vstack([self.bh_array, self.ddh_array])
            kdedens = gaussian_kde(xy)(xy)
            
            sc = axs[0].scatter(self.bh_array, self.ddh_array, c=kdedens)
            if self.num_outf != 0:
                axs[0].scatter(self.bh_out, self.ddh_out, color='green', label='outliers', marker='s')

            axs[0].set_title('Scatterplot')
            axs[0].set_ylabel('DDH: mean {} - std {}'.format(round(np.mean(self.ddh_array), 2), round(np.std(self.ddh_array), 2)))
            axs[0].set_xlabel('BH: mean {} - std {}'.format(round(np.mean(self.bh_array), 2), round(np.std(self.bh_array), 2)))
            axs[0].set_ylim([min_val, max_val])
            axs[0].set_xlim([min_val, max_val])
            axs[0].legend(loc='upper left')
            axs[0].grid()
            if self.logscale == True:
                axs[0].set_xscale("log")
                axs[0].set_yscale("log")

            axs[1].hist(bias, bins=20, color='green')
            statsvals = '''
            Mean: {}
            Std: {}
            '''.format(round(np.mean(bias), 2), round(np.std(bias), 2))
            axs[1].annotate(statsvals, xy=(0.6, 0.85), xycoords='axes fraction', color='black')
            axs[1].grid()

            #qqplot
            x_quant = [np.nanquantile(np.array(self.bh_array), q) for q in np.linspace(0,1,self.bh_array.size)]
            y_quant = [np.nanquantile(np.array(self.ddh_array), q) for q in np.linspace(0,1,self.bh_array.size)]
            sc = axs[2].scatter(x_quant, y_quant, c='black')

            axs[2].set_title('QQ-plot')
            axs[2].set_ylabel('DDH')
            axs[2].set_xlabel('BH')
            axs[2].set_ylim([min_val_out, max_val_out])
            axs[2].set_xlim([min_val_out, max_val_out])
            axs[2].legend(loc='upper left')
            axs[2].grid()
            #cbar = fig.colorbar(sc, ax=axs[1])

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
            axs[3].annotate(statsvals, xy=(0.4, 0.0), xycoords='axes fraction', color='black')
            bh_y = np.linspace(0, 1, len(self.bh_array), endpoint=True)
            lt_y = np.linspace(0, 1, len(self.ddh_array), endpoint=True)
            axs[3].plot(np.sort(self.bh_array), bh_y, label='BH', color='blue')
            axs[3].plot(np.sort(self.ddh_array), lt_y, label='DDH', color='red')
            axs[3].set_title('KS test - eCDF')
            axs[3].set_ylim([0,1])
            axs[3].set_xlim([min_val,max_val])
            axs[3].legend(loc='upper right')
            axs[3].grid()

            #hard plot
            sort_idxs = np.argsort(abs_rel_bias)
            ordered_rel_bias = abs_rel_bias[sort_idxs]
            hard = ordered_rel_bias/2
            sum_hard = np.cumsum(ordered_rel_bias)
            sum_hard = (sum_hard - np.min(sum_hard)) / (np.max(sum_hard) - np.min(sum_hard)) * 100
            perc_samples = np.array([100*(i+1)/len(sort_idxs) for i in range(len(sort_idxs))])

            axs[4].scatter(perc_samples, sum_hard, c='blue', marker='o', s=1)

            f90 = perc_samples >= 90
            gt90 = perc_samples[f90][0]
            hard_gt90 = sum_hard[f90][0]
            lt90 = perc_samples[~f90][-1]
            hard_lt90 = sum_hard[~f90][-1]
            p90 = interpolation([[lt90, hard_lt90], [gt90, hard_gt90]], 90)
            axs[4].vlines(x=90, ymin=0, ymax=p90, color='red', linestyles='--')
            axs[4].hlines(y=p90, xmin=0, xmax=90, color='red', linestyles='--')

            axs[4].set_xlabel('Samples proportion')
            axs[4].set_ylabel('Relative error')
            axs[4].set_title('Hard plot')
            axs[4].grid()
            axs[4].set_xlim(0,100)
            axs[4].set_xticks([i*10 for i in range(11)])
            axs[4].set_ylim(0, np.max(sum_hard))

            #saving
            plt.suptitle('{} - DDH-BH analysis'.format(self.var_name))

            plt.savefig(outfl, dpi=300, facecolor='white', bbox_inches='tight')

            #return mean_rel_bias, max_bias, r_value
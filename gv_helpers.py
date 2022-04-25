import math
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

def boxplots(vars_array, vars_names_array, cat, title, outfl):
    n_var = len(vars_array)
    n_lines = math.ceil(n_var/3)
    fig, axs = plt.subplots(n_lines, 3, figsize=(15,15))

    axs = axs.flatten()

    unique_cats = np.unique(cat)

    for i, v in enumerate(vars_array):
        print('Processing {}'.format(vars_names_array[i]))

        fdef = np.isfinite(v)
        v = v[fdef]
        cat_aux = cat[fdef]
        
        axs[i].boxplot([v[cat_aux==c] for c in unique_cats], labels=unique_cats)
        axs[i].set_title(vars_names_array[i])
        axs[i].grid()

    iidx = n_lines*3-(n_lines*3-len(vars_array))
    axs_to_remove = np.arange(iidx, n_lines*3)
    for i in axs_to_remove:
        axs[i].set_visible(False)

    fig.suptitle(title, y=1.0)

    plt.tight_layout()

    plt.savefig(outfl, bbox_inches='tight', facecolor='white')

def cat1_in_cat2(cat1, cat2, title, outfl='cat1incat2.png'):
    fig = plt.figure(figsize=(12,8))

    x_labels = np.unique(cat1)
    x_ticks = [i for i in range(len(x_labels))]

    u_cats2 = np.unique(cat2)

    heights_sum = np.zeros(len(x_labels))

    for idx, i in enumerate(u_cats2):
        cat2_filter = cat2 == i
        
        heights = []
        
        for j in x_labels:
            cat1_filter = cat1 == j
            both = np.logical_and(cat2_filter, cat1_filter)
            heights.append(np.sum(both))
        
        plt.bar(x_ticks, heights, label=i, bottom=heights_sum)
        heights_sum = heights_sum + np.array(heights)

    plt.xticks(x_ticks, x_labels, rotation='vertical')
    plt.ylabel('# samples')
    plt.title(title)
    plt.legend()

    plt.savefig(outfl, bbox_inches='tight', facecolor='white')

def plt_cluster_metrics(Tdata, k, outfl):
    ch_scores_km = [] 
    db_scores_km = [] 
    sl_scores_km = [] 
    inertia_km = []

    ch_scores_h = [] 
    db_scores_h = [] 
    sl_scores_h = [] 
    
    for i in range(2,k): 
        cluster = KMeans(n_clusters=i, algorithm='full')
        cluster.fit(Tdata) 
        labels = cluster.labels_
        ch_scores_km.append(calinski_harabasz_score(Tdata, labels))
        sl_scores_km.append(silhouette_score(Tdata, labels))
        db_scores_km.append(davies_bouldin_score(Tdata, labels))
        inertia_km.append(cluster.inertia_)
            
        cluster = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='ward')
        cluster.fit(Tdata)  
        labels = cluster.labels_
        ch_scores_h.append(calinski_harabasz_score(Tdata, labels))
        sl_scores_h.append(silhouette_score(Tdata, labels))
        db_scores_h.append(davies_bouldin_score(Tdata, labels))

    x = [i for i in range(2,k)]
    fig, axs = plt.subplots(2, 2, figsize=(15,7.5))
    axs[0][0].set_ylabel('Calinski-Harabasz')
    axs[0][0].set_xlabel('# clusters')
    axs[0][0].set_xticks(x)
    axs[0][0].plot(x, ch_scores_km, label='KMeans', marker='o', markersize=5)
    axs[0][0].plot(x, ch_scores_h, label='Hierarchical', marker='o', markersize=5)
    axs[0][0].grid()
    axs[0][0].legend()

    axs[0][1].set_ylabel('Davies-Bouldin')
    axs[0][1].set_xlabel('# clusters')
    axs[0][1].set_xticks(x)
    axs[0][1].plot(x, db_scores_km, label='KMeans', marker='o', markersize=5)
    axs[0][1].plot(x, db_scores_h, label='Hierarchical', marker='o', markersize=5)
    axs[0][1].grid()


    axs[1][0].set_ylabel('Silhouette')
    axs[1][0].set_xlabel('# clusters')
    axs[1][0].set_xticks(x)
    axs[1][0].plot(x, sl_scores_km, label='KMeans', marker='o', markersize=5)
    axs[1][0].plot(x, sl_scores_h, label='Hierarchical', marker='o', markersize=5)
    axs[1][0].grid()


    axs[1][1].set_ylabel('KMeans Inertia')
    axs[1][1].set_xlabel('# clusters')
    axs[1][1].set_xticks(x)
    axs[1][1].plot(x, inertia_km, label='KMeans', marker='o', markersize=5)
    axs[1][1].grid()

    plt.tight_layout()

    plt.savefig(outfl, bbox_inches='tight', facecolor='white')

def confusion_matrix_plot(clf, y_true, y_pred, title, outfl):
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    fig, axs = plt.subplots(1,1, figsize=(8,8))

    acc = round(accuracy_score(y_true, y_pred), 2)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    axs.set_title('{} - acc {} - n {}'.format(title, acc, len(y_true)))

    disp.plot(ax=axs, cmap='Blues', colorbar=False)

    plt.savefig(outfl, facecolor='white', bbox_inches='tight')

def corrmat(df, fsize, outfl):

    f = plt.figure(figsize=fsize)
    correlations = df.corr().values
    plt.matshow(correlations, fignum=f.number, cmap='viridis')
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=90)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    #cb = plt.colorbar()
    #cb.ax.tick_params(labelsize=14)

    for idxi, i in enumerate(range(correlations.shape[0])):
            for idxj, j in enumerate(range(correlations.shape[1])):
                plt.text(idxj, idxi, round(correlations[idxi, idxj], 2), ha="center", va="center", color="black")
                plt.text(idxj, idxi, round(correlations[idxi, idxj], 2), ha="center", va="center", color="black")
    
    plt.savefig(outfl, bbox_inches='tight', facecolor='white')

def hist_cat(cat, title, outfl):
    fig = plt.figure(figsize=(12,8))

    defined_filter = [isinstance(x, str) for x in cat]
    cat = cat[defined_filter]

    x_labels = np.unique(cat)
    x_ticks = [i for i in range(len(x_labels))]

    heights = [np.sum(cat==i) for i in x_labels]

    plt.bar(x_ticks, heights)

    for x, y in zip(x_ticks, heights):
        plt.annotate(str(y), (x, y), ha='center')

    plt.xticks(x_ticks, x_labels, rotation='vertical')
    plt.ylabel('# samples')
    plt.title(title)
    plt.grid()

    plt.savefig(outfl, bbox_inches='tight', facecolor='white')

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

class PCA_analysis:
    def __init__(self, variables_array, variables_names_array):
        self.variable_names_array = variables_names_array

        X = np.array(variables_array)

        sc = StandardScaler()
        Xt = sc.fit_transform(X)

        pca = PCA()
        pcs = pca.fit_transform(Xt)

        self.pca = pca
        self.pcs = pcs

    def plot(self, outdir):
        fix, axs = plt.subplots(1,3,figsize=(21,7))

        #ploting contributions
        pcs_names = ['PC{}'.format(i+1) for i in range(self.pca.n_components_)]
        ticks = np.arange(len(pcs_names))

        axs[0].barh(ticks, np.flip(self.pca.explained_variance_ratio_*100), tick_label=np.flip(pcs_names))
        axs[0].set_xlabel('PC variance contribution')
        axs[0].grid()

        #ploting loadinggs
        axs[1].imshow(self.pca.components_, cmap='bwr')
        axs[1].set_xticks([i for i in range(len(self.variable_names_array))], labels=self.variable_names_array, rotation='vertical')
        axs[1].set_yticks([i for i in range(self.pca.n_components_)], labels=['PC{}'.format(i) for i in range(1,self.pca.n_components_+1)], rotation='horizontal')
        for idxi, i in enumerate(range(self.pca.components_.shape[0])):
            for idxj, j in enumerate(range(self.pca.components_.shape[1])):
                axs[1].text(idxj, idxi, round(self.pca.components_[idxi, idxj], 2), ha="center", va="center", color="black")
                axs[1].text(idxj, idxi, round(self.pca.components_[idxi, idxj], 2), ha="center", va="center", color="black")

        #plotting scatter
        pc1, pc2 = self.pcs[:, 0], self.pcs[:, 1]
        rho = np.corrcoef(pc1, pc2)
        xy = np.vstack([pc1,pc2])
        kdedens = gaussian_kde(xy)(xy)
        axs[2].scatter(pc1, pc2, c=kdedens)
        axs[2].set_xlabel('PC1')
        axs[2].set_ylabel('PC2')
        axs[2].grid()
        axs[2].annotate('rho: {}'.format(rho[1][0].round(2)), xy=(0.8, 0.05), xycoords='axes fraction', color='gray')

        
        plt.tight_layout()
        plt.savefig(outdir+'PCA_analysis.png', bbox_inches='tight', facecolor='white')

        #ploting circle
        plt.figure(figsize=(15,15))
        #Create a list of 500 points with equal spacing between -1 and 1
        x=np.linspace(start=-1,stop=1,num=500)
        #Find y1 and y2 for these points
        y_positive=lambda x: np.sqrt(1-x**2) 
        y_negative=lambda x: -np.sqrt(1-x**2)
        plt.plot(x,list(map(y_positive, x)), color='gray')
        plt.plot(x,list(map(y_negative, x)),color='gray')

        #Plot smaller circle
        x=np.linspace(start=-0.5,stop=0.5,num=500)
        y_positive=lambda x: np.sqrt(0.5**2-x**2) 
        y_negative=lambda x: -np.sqrt(0.5**2-x**2)
        plt.plot(x,list(map(y_positive, x)), color='gray')
        plt.plot(x,list(map(y_negative, x)),color='gray')

        #Create broken lines
        x=np.linspace(start=-1,stop=1,num=30)
        plt.scatter(x,[0]*len(x), marker='_',color='gray')
        plt.scatter([0]*len(x), x, marker='|',color='gray')

        #Define color list
        colors = ['blue', 'red', 'green', 'black', 'purple', 'brown']
        if len(self.pca.components_[0]) > 6:
            colors=colors*(int(len(self.pca.components_[0])/6)+1)
            
        add_string=""
        for i in range(len(self.pca.components_[0])):
            xi=self.pca.components_[0][i]
            yi=self.pca.components_[1][i]
            plt.arrow(0,0, 
                    dx=xi, dy=yi, 
                    head_width=0.03, head_length=0.03, 
                    color=colors[i], length_includes_head=True)
            add_string=f" ({round(xi,2)} {round(yi,2)})"
            plt.text(self.pca.components_[0, i], 
                    self.pca.components_[1, i] , 
                    s=self.variable_names_array[i] + add_string )
            
        plt.xlabel(f"Component 1 ({round(self.pca.explained_variance_ratio_[0]*100,2)}%)")
        plt.ylabel(f"Component 2 ({round(self.pca.explained_variance_ratio_[1]*100,2)}%)")
        plt.grid()
        plt.savefig(outdir+'PCAcircle.png', bbox_inches='tight', facecolor='white')

def proportional_effect(vars_array, var_names_array, cat, title, outfl):
    n_var = len(vars_array)
    n_lines = math.ceil(n_var/3)
    fig, axs = plt.subplots(n_lines, 3, figsize=(15,15))

    axs = axs.flatten()

    unique_cats = np.unique(cat)

    for i, v in enumerate(vars_array):
        for j, d in enumerate(unique_cats):

            catf = cat == d

            mean = np.nanmean(v[catf])
            std = np.nanstd(v[catf])

            axs[i].scatter([mean], [std], label=d, s=30, marker='s')

        axs[i].set_title(var_names_array[i])
        axs[i].grid()
        axs[i].legend()
        axs[i].set_xlabel('Mean')
        axs[i].set_ylabel('Std dev')

    iidx = n_lines*3-(n_lines*3-len(vars_array))
    axs_to_remove = np.arange(iidx, n_lines*3)
    for i in axs_to_remove:
        axs[i].set_visible(False)

    fig.suptitle(title, y=1.0)

    plt.tight_layout()

    plt.savefig(outfl, bbox_inches='tight', facecolor='white')

def scatter_plot(x, y, ax, c=None):
    if c is None:
        xy = np.vstack([x, y])
        c = gaussian_kde(xy)(xy)

    corr = round(np.corrcoef([x, y])[0,1], 2)

    ax.scatter(x, y, c=c, s=1, cmap='jet')
    ax.set_title('rho: {}'.format(corr))


def scatter_matrix(vars_array, vars_name, title, outfl, nmax=None, cat=None):
    if nmax is not None:
        idxs = np.arange(len(vars_array[0]))
        ridxs = np.random.choice(idxs, size=nmax, replace=False)
        vars_array = [i[ridxs] for i in vars_array]
    else:
        pass

    data = np.array(vars_array)
    numvars, numdata = data.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(12,12))

    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i,j), (j,i)]:

            scatter_plot(data[x], data[y], axes[x,y], cat)
            axes[x,y].grid()
            axes[x,y].set_xlabel(vars_name[x])
            axes[x,y].set_ylabel(vars_name[y])

    for i, label in enumerate(vars_name):
        axes[i,i].hist(data[i], color='green')
        axes[i,i].set_title(label)


    plt.tight_layout()
    plt.savefig(outfl, facecolor='white', bbox_inches='tight')

def tdscatter(x, y, z, c, zex, outfl):

    cdef = np.isfinite(c)
    c = c[cdef]
    x, y, z = x[cdef], y[cdef], z[cdef]
 
    fig = plt.figure(figsize = (15, 15))
    ax = plt.axes(projection ="3d")
    
    ax.set_box_aspect((np.ptp(x), np.ptp(y), zex*np.ptp(z)))  # aspect ratio is 1:1:1 in data space

    ax.scatter3D(x, y, z, c = c, s=2, cmap='jet')

    plt.tight_layout()
    
    plt.savefig(outfl, bbox_inches='tight', facecolor='white')

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
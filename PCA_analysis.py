from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt

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
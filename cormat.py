import matplotlib.pyplot as plt
import matplotlib.colors as colors

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
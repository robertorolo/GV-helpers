from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

import matplotlib.pyplot as plt

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
    axs[0][0].set_title('Higher is better')
    axs[0][0].set_ylabel('Calinski-Harabasz')
    axs[0][0].set_xlabel('# clusters')
    axs[0][0].set_xticks(x)
    axs[0][0].plot(x, ch_scores_km, label='KMeans', marker='o', markersize=5)
    axs[0][0].plot(x, ch_scores_h, label='Hierarchical', marker='o', markersize=5)
    axs[0][0].grid()
    axs[0][0].legend()

    axs[0][1].set_title('CLoser to zero is better')
    axs[0][1].set_ylabel('Davies-Bouldin')
    axs[0][1].set_xlabel('# clusters')
    axs[0][1].set_xticks(x)
    axs[0][1].plot(x, db_scores_km, label='KMeans', marker='o', markersize=5)
    axs[0][1].plot(x, db_scores_h, label='Hierarchical', marker='o', markersize=5)
    axs[0][1].grid()


    axs[1][0].set_title('Higher is better')
    axs[1][0].set_ylabel('Silhouette')
    axs[1][0].set_xlabel('# clusters')
    axs[1][0].set_xticks(x)
    axs[1][0].plot(x, sl_scores_km, label='KMeans', marker='o', markersize=5)
    axs[1][0].plot(x, sl_scores_h, label='Hierarchical', marker='o', markersize=5)
    axs[1][0].grid()


    axs[1][1].set_title('Use the elbow method')
    axs[1][1].set_ylabel('KMeans Inertia')
    axs[1][1].set_xlabel('# clusters')
    axs[1][1].set_xticks(x)
    axs[1][1].plot(x, inertia_km, label='KMeans', marker='o', markersize=5)
    axs[1][1].grid()

    plt.tight_layout()

    plt.savefig(outfl, bbox_inches='tight', facecolor='white')
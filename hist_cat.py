import numpy as np
import matplotlib.pyplot as plt

def hist_cat(cat, title, outfl):
    fig = plt.figure(figsize=(12,8))

    x_labels = np.unique(cat)
    x_ticks = [i for i in range(len(x_labels))]

    heights = [np.sum(cat==i) for i in x_labels]

    plt.bar(x_ticks, heights)

    plt.xticks(x_ticks, x_labels, rotation='vertical')
    plt.ylabel('# samples')
    plt.title(title)
    plt.legend()
    plt.grid()

    plt.savefig(outfl, bbox_inches='tight', facecolor='white')
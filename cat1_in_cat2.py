import numpy as np
import matplotlib.pyplot as plt

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
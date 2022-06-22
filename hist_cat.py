import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def hist_cat(cat, title, outfl, cmap='jet'):
    fig = plt.figure(figsize=(12,8))

    #defined_filter = [isinstance(x, str) for x in cat]
    #cat = cat[defined_filter]

    x_labels = np.unique(cat)
    x_ticks = [i for i in range(len(x_labels))]
    cmap = matplotlib.cm.get_cmap(cmap, len(x_labels))
    colors = [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]

    heights = [np.sum(cat==i) for i in x_labels]

    plt.bar(x_ticks, heights, color=colors)

    for x, y in zip(x_ticks, heights):
        plt.annotate(str(y), (x, y), ha='center')

    plt.xticks(x_ticks, x_labels, rotation='vertical')
    plt.ylabel('# samples')
    plt.title(title)
    plt.grid()

    plt.savefig(outfl, bbox_inches='tight', facecolor='white')
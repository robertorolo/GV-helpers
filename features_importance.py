import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

def plot_features_importance(model, X_test, varnames, y_test, outfl):
    fig, axs = plt.subplots(2, 1, figsize=(10,6))
    
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=varnames)
    
    forest_importances.plot.bar(yerr=std, ax=axs[0])
    axs[0].set_title("Feature importances using mean decrease")
    axs[0].set_ylabel("Mean decrease")
    axs[0].grid()
    
    result = permutation_importance(model, X_test, y_test, scoring='accuracy', n_repeats=10)
    forest_importances = pd.Series(result.importances_mean, index=varnames)

    forest_importances.plot.bar(yerr=result.importances_std, ax=axs[1])
    axs[1].set_title("Feature importances using permutation")
    axs[1].set_ylabel("Acuracy decrease")
    axs[1].grid()

    fig.tight_layout()
    fig.savefig(outfl, facecolor='white')

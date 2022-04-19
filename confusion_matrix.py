from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def confusion_matrix_plot(clf, y_true, y_pred, title, outfl):
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    fig, axs = plt.subplots(1,1, figsize=(8,8))

    acc = round(accuracy_score(y_true, y_pred), 2)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    axs.set_title('{} - acc {}'.format(title, acc))

    disp.plot(ax=axs, cmap='Blues', colorbar=False)

    plt.savefig(outfl, facecolor='white', bbox_inches='tight')
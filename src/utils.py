import matplotlib.pyplot as plt


def plot_confusion_matrix(
    cm, target_names, title="Confusion matrix", cmap=None, normalize=False, ft=16
):
    """
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap("Blues")

    fig = plt.figure(figsize=(16, 16))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=ft)
    # plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90, fontsize=ft)
        plt.yticks(tick_marks, target_names, fontsize=ft)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(
                j,
                i,
                "{:0.4f}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=ft
            )
        else:
            plt.text(
                j,
                i,
                "{:,}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=ft
            )

    # plt.tight_layout()
    plt.ylabel("True label", fontsize=ft)
    plt.xlabel(
        "Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(
            accuracy, misclass), fontsize=ft
    )
    return fig

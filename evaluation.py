import os.path

from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import torch

import utils
from dataset import LABELS_USED


def accuracy_score(y_true, y_pred, class_counts=None):
    rounded_pred = utils.get_class_prediction(y_pred)
    correct = (rounded_pred == y_true).float()
    label_accuracy = torch.mean(correct, dim=0)
    if class_counts is not None:
        weights = class_counts / torch.sum(class_counts)
        mean_accuracy = torch.sum(label_accuracy * weights)
    else:
        mean_accuracy = torch.mean(correct)

    return mean_accuracy, label_accuracy


def plot_confusion_matrix(confusion_matrix, save_name=None):
    disp = ConfusionMatrixDisplay(confusion_matrix)
    disp.plot()
    if save_name:
        plt.imsave("evaluation/" + save_name + ".png")
    plt.show()


if __name__ == "__main__":
    path_to_test_pkl = "trained_models/nih_chest_densenet_pretrained_new2_test.pkl"

    with open(path_to_test_pkl, "rb") as f:
        log = pkl.load(f)

    y_true = np.vstack(log['target']).astype(int)
    y_pred = np.vstack(log['prediction']).astype(int)

    # add "no finding" as class
    no_finding_vector_true = (y_true.sum(axis=1) == 0).astype(int)
    no_finding_vector_pred = (y_pred.sum(axis=1) == 0).astype(int)

    y_true = np.hstack((y_true, no_finding_vector_true.reshape(-1, 1)))
    y_pred = np.hstack((y_pred, no_finding_vector_pred.reshape(-1, 1)))

    accuracy_no_findings = np.equal(no_finding_vector_true, no_finding_vector_pred)
    accuracy = np.vstack(log['label accuracy'])
    accuracy = np.hstack((accuracy, accuracy_no_findings.reshape(-1, 1))).mean(axis=0)

    precision = []
    recall = []
    f1_score = []

    confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)

    for matrix in confusion_matrix:
        tp, tn, fp, fn = matrix.ravel()

        precision.append(tp / (tp + fp))
        recall.append(tp / (tp + fn))
        f1_score.append((2 * precision[-1] * recall[-1]) / (precision[-1] + recall[-1]))

    macro_avg_precision = np.asarray(precision).mean()
    micro_avg_precision = confusion_matrix[:, 0, 0] / (confusion_matrix[:, 0, 0] + confusion_matrix[:, 0, 1])

    save_filename = os.path.basename(path_to_test_pkl).split("_test")[0]

    eval_metrics = {
        "label": LABELS_USED,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1 score": f1_score,
        "macro_avg_precision": macro_avg_precision,
        "micro_avg_precision": micro_avg_precision
    }

    with open(os.path.dirname(path_to_test_pkl) + "/" + save_filename + "_evaluation.pkl", "wb") as f:
        pkl.dump(eval_metrics, f)

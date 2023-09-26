import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

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


def classification_metrics(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, (tp + fn), out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1_score = np.divide((2 * precision * recall), (precision + recall),
                         out=np.zeros_like(2 * precision * recall),
                         where=(precision + recall) != 0)
    accuracy = np.divide(tn + tp, tn + fp + fn + tp,
                         out=np.zeros_like(tn + tp),
                         where=(tn + fp + fn + tp) != 0)

    return precision, recall, f1_score, accuracy


def test_evaluation(path_to_test_pkl):
    """
    Takes the logs which were saved by test_base_model.py or base_model_cotta.py and computes several metrics.
    :param path_to_test_pkl: Path to the pickle file which contains the test / cotta logs.
    """
    with open(path_to_test_pkl, "rb") as f:
        log = pkl.load(f)

    y_true = np.vstack(log['target']).astype(int)
    y_pred = np.vstack(log['prediction']).astype(int)

    # add "no finding" as class
    no_finding_vector_true = (y_true.sum(axis=1) == 0).astype(int)
    no_finding_vector_pred = (y_pred.sum(axis=1) == 0).astype(int)

    y_true = np.hstack((y_true, no_finding_vector_true.reshape(-1, 1)))
    y_pred = np.hstack((y_pred, no_finding_vector_pred.reshape(-1, 1)))

    precision = []
    recall = []
    f1_score = []
    accuracy = []

    label_confusion_matrix = np.zeros((8, 2, 2))
    total_confusion_matrix = np.zeros((2, 2))

    for prediction, target in zip(y_pred, y_true):
        confusion_matrix = multilabel_confusion_matrix(np.asarray(target).reshape(1, -1),
                                                       np.asarray(prediction).reshape(1, -1)).astype(float)
        total_confusion_matrix += confusion_matrix.sum(axis=0)
        label_confusion_matrix += confusion_matrix

        prec, rec, f1, acc = classification_metrics(confusion_matrix.sum(axis=0))

        precision.append(prec.item())
        recall.append(rec.item())
        f1_score.append(f1.item())
        accuracy.append(acc.item())

    metrics = {
        "target": log["target"],
        "prediction": log["prediction"],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1 score": f1_score
    }

    avg_metrics = {
        "accuracy": np.mean(np.vstack(accuracy), axis=0),
        "precision": np.mean(np.vstack(precision), axis=0),
        "recall": np.mean(np.vstack(recall), axis=0),
        "f1 score": np.mean(np.vstack(f1_score), axis=0),
        "total confusion matrix": total_confusion_matrix
    }

    label_precision = []
    label_recall = []
    label_f1_score = []
    label_accuracy = []

    for matrix in label_confusion_matrix:
        prec, rec, f1, acc = classification_metrics(matrix)
        label_precision.append(prec.item())
        label_recall.append(rec.item())
        label_f1_score.append(f1.item())
        label_accuracy.append(acc.item())

    label_metrics = {
        "labels": LABELS_USED,
        "accuracy": label_accuracy,
        "precision": label_precision,
        "recall": label_recall,
        "f1 score": label_f1_score,
        "label confusion matrix": label_confusion_matrix
    }

    if "test" in path_to_test_pkl:
        save_filename = os.path.basename(path_to_test_pkl).split("_test")[0]
    else:
        metrics["batch num"] = log["batch num"]
        save_filename = os.path.basename(path_to_test_pkl).split(".")[0]

    with open("results/" + save_filename + "_evaluation.pkl", "wb") as f:
        pkl.dump(metrics, f)
    with open("results/" + save_filename + "_evaluation.csv", "wb") as f:
        pd.DataFrame(metrics).to_csv(f)
    with open("results/" + save_filename + "_avg_evaluation.pkl", "wb") as f:
        pkl.dump(avg_metrics, f)
    with open("results/" + save_filename + "_label_evaluation.pkl", "wb") as f:
        pkl.dump(label_metrics, f)


if __name__ == "__main__":
    test_evaluation("trained_models/chex_densenet_cotta_batch2.pkl")

import torch
import utils

def accuracy(y_true, y_pred, class_counts=None):
    rounded_pred = utils.get_class_prediction(y_pred)
    correct = (rounded_pred == y_true).float()
    label_accuracy = torch.mean(correct, dim=0)
    if class_counts is not None:
        weights = class_counts / torch.sum(class_counts)
        mean_accuracy = torch.sum(label_accuracy * weights)
    else:
        mean_accuracy = torch.mean(correct)
    return mean_accuracy, label_accuracy
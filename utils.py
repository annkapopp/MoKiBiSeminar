import torch


def constrain_prediction(output, class_prediction):
    no_findings_indices = torch.where(class_prediction[:, -1] == 1)[0]
    class_prediction[no_findings_indices, :-1] = 0
    output[no_findings_indices, :-1] = 0
    return output, class_prediction


def get_class_prediction(output):
    return (output > 0.5).int()

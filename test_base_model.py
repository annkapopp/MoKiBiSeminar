import torch
from torch import nn, optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle as pkl
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, precision_score, recall_score, f1_score
import torchxrayvision as txrv
import torch.nn.functional as F

from base_models import ResNet18, DenseNet121
import dataset
from evaluation import accuracy_score
import utils
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(model, model_name, dataloader):
    log = {
        'image name': [],
        'target': [],
        'prediction': [],
        "confusion matrix": [],
        "mean accuracy_score": [],
        "label accuracy_score": [],
        "recall": [],
        "precision": [],
        "f1 score": []
    }
    with torch.no_grad():
        model.eval()

        for input, _, target, image_name in tqdm(dataloader, position=0, leave=True):
            input = input.to(DEVICE)
            target = target.to(DEVICE)
            output = model(input)

            # output = F.sigmoid(output)

            class_prediction = utils.get_class_prediction(output)
            _, class_prediction = utils.constrain_prediction(output, class_prediction)
            class_prediction = class_prediction.detach().cpu().numpy()

            log["target"].append(target.detach().cpu().numpy())
            log["prediction"].append(class_prediction)

            # print(class_prediction, target)

            mean_accuracy, label_accuracy = accuracy_score(target, output, class_counts=dataloader.dataset
                                                           .__target_label_counts__().to(DEVICE))
            log["mean accuracy_score"].append(mean_accuracy.detach().cpu().numpy())
            log["label accuracy_score"].append(label_accuracy.detach().cpu().numpy())

            target = target.detach().cpu().numpy()
            log["confusion matrix"].append(multilabel_confusion_matrix(target, class_prediction))
            log["precision"].append(precision_score(target, class_prediction, average="micro"))
            log["recall"].append(recall_score(target, class_prediction, average="micro", zero_division=0.0))
            log["f1 score"].append(f1_score(target, class_prediction, average="micro"))

            log["image name"].append(image_name)

        with open("trained_models/" + model_name + "_test.pkl", "wb") as f:
            pkl.dump(log, f)


if __name__ == "__main__":
    print("device: ", DEVICE)

    dataset_train = dataset.NIHChestXRayDataset("NIH Chest Xray", mode="test")

    dataloader = DataLoader(dataset_train, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)

    model = DenseNet121(pretrained=False)
    # model = txrv.models.DenseNet(weights="densenet121-res224-nih", apply_sigmoid=False)
    model_name = "nih_chest_densenet_pretrained_new2"
    model.load_state_dict(torch.load("trained_models/" + model_name + ".pt"))

    test(model.to(DEVICE), model_name, dataloader)

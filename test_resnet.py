import torch
from torch import nn, optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle as pkl
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, precision_score, recall_score, f1_score

from resnet import ResNet18
import dataset
from evaluation_metrics import accuracy
import utils

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def test(model, model_name, num_epochs, train_dataloader, val_dataloader, criterion, optimizer, scheduler):
    log = {
        'image name': [],
        "confusion matrix": [],
        "mean accuracy": [],
        "recall": [],
        "precision": [],
        "f1": []
    }
    with torch.inference_mode():
        model.eval()

        running_loss = 0.0
        running_accuracy = 0.0

        for input, _, target, image_name in val_dataloader:
            input = input.to(DEVICE)
            target = target.numpy()
            output = model(input)

            class_prediction = utils.get_class_prediction(output)
            _, class_prediction = utils.constrain_prediction(output, class_prediction)
            class_prediction = class_prediction.detach().numpy().cpu()

            log["confusion matrix"].append(multilabel_confusion_matrix(target, class_prediction))
            log["precision"].append(precision_score(target, class_prediction, average="micro"))
            log["recall"].append(precision_score(target, class_prediction, average="micro"))
            log["f1_score"].append(precision_score(target, class_prediction, average="micro"))

            mean_accuracy, _ = accuracy(target, output, class_counts=train_dataloader.dataset.__label_counts__().to(DEVICE))
            log["mean accuracy"].append(mean_accuracy.detach().cpu().numpy())

            log["image name"].append(image_name)

        # f1 = f1_score(y_true=target.detach().cpu().numpy(), y_pred=output.detach().cpu().numpy(),
                      # average='micro')

        with open("E:\\trained_models\\" + model_name + "_test.pkl", "wb") as f:
            pkl.dump(log, f)

        tqdm.write('Epoch {} (valid) -- loss: {:.4f} acc: {:.4f}'.format(running_loss,
                                                                         running_accuracy))


if __name__ == "__main__":
    print("device: ", DEVICE)

    dataset_train = dataset.NIHChestXRayDataset("E:\\NIH Chest Xray", mode="train")
    dataset_val = dataset.NIHChestXRayDataset("E:\\NIH Chest Xray", mode="val")

    dataloader_train = DataLoader(dataset_train, batch_size=24, shuffle=True, pin_memory=True, num_workers=4)
    dataloader_val = DataLoader(dataset_val, batch_size=24, shuffle=False, pin_memory=True, num_workers=4)

    label_counts = dataset_train.__label_counts__()
    weights = np.sqrt(1 / label_counts)
    weights /= weights.mean()

    model = ResNet18().to(DEVICE)
    lr = 0.01
    criterion = nn.BCELoss(weight=weights.to(DEVICE, dtype=torch.float))
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    train(model, "nih_chest_resnet18", 100, dataloader_train, dataloader_val,
          criterion, optimizer, scheduler)

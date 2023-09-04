import torch
from torch import nn, optim
import numpy as np
from tqdm.notebook import trange
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from sklearn.metrics import balanced_accuracy_score, f1_score, average_precision_score

from resnet import ResNet18
import dataset
import utils

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, model_name, num_epochs, train_dataloader, val_dataloader, criterion, optimizer, scheduler):
    losses_train = []
    accuracies_train = []
    f1s_train = []
    losses_val = []
    accuracies_val = []
    f1s_val = []

    for epoch in trange(num_epochs, unit="epochs"):

        # training
        running_loss = 0.0
        running_accuracy = 0.0

        model.train()

        for input, target in train_dataloader:
            input = input.to(DEVICE)
            target = target.long().to(DEVICE)
            optimizer.zero_grad()

            output = model(input)
            
            loss = criterion(output, target)
            loss.backward()

            running_loss += loss.item()

            prediction = (torch.sigmoid(output) > 0.5).int()
            prediction = torch.round(prediction).squeeze().tolist()
            prediction = [int[p] for p in prediction]

            running_accuracy = balanced_accuracy_score(y_true=target, y_pred=prediction,
                                                       sample_weight=criterion.pos_weight)
            optimizer.step()


        running_loss /= (len(train_dataloader) * dataloader_train.batch_size)
        running_accuracy /= (len(train_dataloader) * dataloader_train.batch_size)
        f1 = f1_score(y_true=target, y_pred=output, average='micro')

        losses_train.append(running_loss)
        accuracies_train.append(running_accuracy)
        f1s_train.append(f1)

        # output
        if epoch % 20 == 0:
            tqdm.write('Epoch {} (train) -- loss: {:.2f} acc: {:.2f}'.format(epoch, running_loss, running_accuracy))


        # validation
        with torch.no_grad():
            model.eval()

            running_loss = 0.0
            running_accuracy = 0.0

            for input, target in val_dataloader:
                input = input.to(DEVICE)
                target = target.long().to(DEVICE)
                output = model(input)
                loss = criterion(output, target)

                running_loss += loss.item()

                prediction = (torch.sigmoid(output) > 0.5).int()
                prediction = torch.round(prediction).squeeze().tolist()
                prediction = [int[p] for p in prediction]

                running_accuracy = balanced_accuracy_score(y_true=target, y_pred=prediction,
                                                           sample_weight=criterion.pos_weight)

            running_loss /= (len(val_dataloader) * dataloader_val.batch_size)
            running_accuracy /= (len(val_dataloader) * dataloader_val.batch_size)
            f1 = f1_score(y_true=target, y_pred=output, average='micro')

            scheduler.step(running_loss)

            losses_val.append(running_loss)
            accuracies_val.append(running_accuracy)
            f1s_val.append(f1)

            # output
            if epoch % 20 == 0:
                tqdm.write('Epoch {} (valid) -- loss: {:.2f} acc: {:.2f}, f1: {:.2f}'.format(epoch, running_loss,
                                                                                             running_accuracy, f1))

    torch.save(model.state_dict(), "/Volumes/Temp/trained_models/" + model_name + ".pt")


if __name__ == "__main__":

    dataset_train = dataset.NIHChestXRayDataset("/Volumes/Temp/NIH Chest Xray", mode="train")
    dataset_val = dataset.NIHChestXRayDataset("/Volumes/Temp/NIH Chest Xray", mode="val")

    dataloader_train = DataLoader(dataset_train, batch_size=2, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False)

    label_counts = dataloader_train.dataset.data['target_vector'].sum().astype(float)
    weights = np.sqrt(1 / label_counts)
    weights /= weights.mean()

    model = ResNet18().to(DEVICE)
    lr = 0.001
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5)

    train(model.double(), "nih_chest_resnet18", 100, dataloader_train, dataloader_val,
          criterion, optimizer, scheduler)

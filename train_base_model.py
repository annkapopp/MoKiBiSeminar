import torch
from torch import nn, optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle as pkl
import os

from base_models import ResNet18, DenseNet121
import dataset
from evaluation import accuracy_score
import utils

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def train(model, model_name, num_epochs, train_dataloader, val_dataloader, criterion, optimizer, scheduler):
    log = {
        "loss_train": [],
        "loss_val": [],
        "accuracy_train": [],
        "accuracy_val": [],
        "f1_train": [],
        "f1_val": []
    }

    for epoch in tqdm(range(num_epochs), position=0, leave=True):
        # training
        running_loss = 0.0
        running_accuracy = 0.0

        model.train()

        for input, _, target, _ in train_dataloader:
            input = input.to(DEVICE)
            target = target.to(DEVICE)
            optimizer.zero_grad()

            output = model(input)
            class_prediction = utils.get_class_prediction(output)
            #output, class_prediction = utils.constrain_prediction(output.clone(), class_prediction)

            loss = criterion(output, target.float())
            loss.backward()

            running_loss += loss.item()

            mean_accuracy, _ = accuracy_score(target, output, class_counts=train_dataloader.dataset
                                              .__target_label_counts__().to(DEVICE))
            running_accuracy += mean_accuracy.detach().cpu().numpy()
            optimizer.step()

        running_loss /= (len(train_dataloader) * dataloader_train.batch_size)
        running_accuracy /= len(train_dataloader)

        log["loss_train"].append(running_loss)
        log["accuracy_train"].append(running_accuracy)

        # output
        tqdm.write('Epoch {} (train) -- loss: {:.4f} acc: {:.4f}'.format(epoch, running_loss, running_accuracy))


        # validation
        with (torch.no_grad()):
            model.eval()

            running_loss = 0.0
            running_accuracy = 0.0

            for input, _, target, _ in val_dataloader:
                input = input.to(DEVICE)
                target = target.to(DEVICE)
                output = model(input)

                class_prediction = utils.get_class_prediction(output)

                loss = criterion(output, target.float())
                running_loss += loss.item()

                mean_accuracy, _ = accuracy_score(target, output, class_counts=train_dataloader.dataset
                                                  .__target_label_counts__().to(DEVICE))
                running_accuracy += mean_accuracy.detach().cpu().numpy()

            running_loss /= (len(val_dataloader) * dataloader_val.batch_size)
            running_accuracy /= len(val_dataloader)

            scheduler.step()

            log["loss_val"].append(running_loss)
            log["accuracy_val"].append(running_accuracy)

            # output
            torch.save(model.state_dict(), "trained_models/" + model_name + ".pt")
            with open("trained_models/" + model_name + "_train_log.pkl", "wb") as f:
                pkl.dump(log, f)

            tqdm.write('Epoch {} (valid) -- loss: {:.4f} acc: {:.4f}'.format(epoch, running_loss,
                                                                             running_accuracy))

    torch.save(model.state_dict(), "trained_models/" + model_name + ".pt")
    with open("trained_models/" + model_name + "_train_log.pkl", "wb") as f:
        pkl.dump(log, f)


if __name__ == "__main__":
    print("device: ", DEVICE)

    # dataset_train = dataset.NIHChestXRayDataset("NIH Chest Xray", mode="train")
    # dataset_val = dataset.NIHChestXRayDataset("NIH Chest Xray", mode="val")
    dataset_train = dataset.CheXpertDataset("CheXpert", mode="train")
    dataset_val = dataset.CheXpertDataset("CheXpert", mode="val")

    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, pin_memory=True, num_workers=8)
    dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False, pin_memory=True, num_workers=8)

    label_counts = dataset_train.__target_label_counts__()
    weights = np.sqrt(1 / label_counts)
    weights /= weights.mean()

    model = DenseNet121(pretrained=True).to(DEVICE)
    lr = 0.00001
    criterion = nn.BCELoss(weight=weights.to(DEVICE, dtype=torch.float))
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    train(model, "chex_densenet", 10, dataloader_train, dataloader_val,
          criterion, optimizer, scheduler)

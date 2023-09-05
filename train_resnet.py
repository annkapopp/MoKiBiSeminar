import torch
from torch import nn, optim
import numpy as np
from tqdm.notebook import trange
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from sklearn.metrics import balanced_accuracy_score, f1_score, average_precision_score, accuracy_score
import pickle as pkl
import time

from resnet import ResNet18
import dataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, model_name, num_epochs, train_dataloader, val_dataloader, criterion, optimizer, scheduler):
    log = {
        "loss_train": [],
        "loss_val": [],
        "accuracy_train": [],
        "accuracy_val": [],
        "f1_train": [],
        "f1_val": []
    }

    for epoch in trange(num_epochs, unit="epochs"):
        # training
        running_loss = 0.0
        running_accuracy = 0.0

        model.train()

        # start_load = time.time()
        for input, _, target, _ in train_dataloader:
            # end_load = time.time()
            # print("load time: ", end_load - start_load)

            input = input.to(DEVICE)
            target = target.to(DEVICE)
            optimizer.zero_grad()

            # start_model = time.time()

            output = model(input)

            loss = criterion(output, target.float())
            loss.backward()
            # end_model = time.time()

            #print("model time: ", end_model - start_model)


            running_loss += loss.item()

            prediction = (torch.sigmoid(output) > 0.5).int()
            prediction = torch.round(prediction).squeeze()

            running_accuracy = accuracy_score(y_true=target.detach().cpu().numpy(),
                                                       y_pred=prediction.detach().cpu().numpy())
            optimizer.step()

            # start_load = time.time()



        running_loss /= (len(train_dataloader) * dataloader_train.batch_size)
        running_accuracy /= (len(train_dataloader) * dataloader_train.batch_size)
        # f1 = f1_score(y_true=target.detach().cpu().numpy(), y_pred=output.detach().cpu().numpy(),
                      # average='micro')

        log["loss_train"].append(running_loss)
        log["accuracy_train"].append(running_accuracy)
        # log["f1_train"].append(f1)

        # output
        #if epoch % 5 == 0:
        tqdm.write('Epoch {} (train) -- loss: {:.2f} acc: {:.2f}'.format(epoch, running_loss, running_accuracy))


        # validation
        with (torch.no_grad()):
            model.eval()

            running_loss = 0.0
            running_accuracy = 0.0

            for input, _, target, _ in val_dataloader:
                input = input.to(DEVICE)
                target = target.to(DEVICE)
                output = model(input)
                loss = criterion(output, target.float())

                running_loss += loss.item()

                prediction = (torch.sigmoid(output) > 0.5).int()
                prediction = torch.round(prediction).squeeze()

                running_accuracy = accuracy_score(y_true=target.detach().cpu().numpy(),
                                                  y_pred=prediction.detach().cpu().numpy())

            running_loss /= (len(val_dataloader) * dataloader_val.batch_size)
            running_accuracy /= (len(val_dataloader) * dataloader_val.batch_size)
            # f1 = f1_score(y_true=target.detach().cpu().numpy(), y_pred=output.detach().cpu().numpy(),
                          # average='micro')

            scheduler.step(running_loss)

            log["loss_val"].append(running_loss)
            log["accuracy_val"].append(running_accuracy)
            # log["f1_val"].append(f1)

            # output
            if epoch % 5 == 0:
                torch.save(model.state_dict(), "E:\\trained_models\\" + model_name + ".pt")
                with open("E:\\trained_models\\" + model_name + "_train_log.pkl", "wb") as f:
                    pkl.dump(log, f)

            tqdm.write('Epoch {} (valid) -- loss: {:.2f} acc: {:.2f}'.format(epoch, running_loss,
                                                                             running_accuracy))

    torch.save(model.state_dict(), "E:\\trained_models\\" + model_name + ".pt")
    with open("E:\\trained_models\\" + model_name + "_train_log.pkl", "wb") as f:
        pkl.dump(log, f)


if __name__ == "__main__":
    print("device: ", DEVICE)

    dataset_train = dataset.NIHChestXRayDataset("E:\\NIH Chest Xray", mode="train")
    dataset_val = dataset.NIHChestXRayDataset("E:\\NIH Chest Xray", mode="val")

    dataloader_train = DataLoader(dataset_train, batch_size=24, shuffle=True, pin_memory=True, num_workers=4)
    dataloader_val = DataLoader(dataset_val, batch_size=24, shuffle=False, pin_memory=True, num_workers=4)

    label_counts = np.stack(dataloader_train.dataset.data['one_hot'].values).sum(axis=0).astype(float)
    weights = np.sqrt(1 / label_counts)
    weights /= weights.mean()
    weights = torch.from_numpy(weights)

    model = ResNet18().to(DEVICE)
    lr = 0.001
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights.to(DEVICE, dtype=torch.float))
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5)

    train(model, "nih_chest_resnet18", 100, dataloader_train, dataloader_val,
          criterion, optimizer, scheduler)

import torch
from torch import nn, optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle as pkl

from resnet import ResNet18
import dataset
from evaluation_metrics import accuracy
import utils

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

    for epoch in tqdm(range(num_epochs), position=0, leave=True):
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
            class_prediction = utils.get_class_prediction(output)
            output, class_prediction = utils.constrain_prediction(output.clone(), class_prediction)

            loss = criterion(output, target.float())
            loss.backward()

            running_loss += loss.item()

            mean_accuracy, _ = accuracy(target, output, class_counts=train_dataloader.dataset.__label_counts__().to(DEVICE))
            running_accuracy += mean_accuracy.detach().cpu().numpy()
            optimizer.step()

            # start_load = time.time()

        running_loss /= (len(train_dataloader) * dataloader_train.batch_size)
        running_accuracy /= len(train_dataloader)
        # f1 = f1_score(y_true=target.detach().cpu().numpy(), y_pred=output.detach().cpu().numpy(),
                      # average='micro')

        log["loss_train"].append(running_loss)
        log["accuracy_train"].append(running_accuracy)
        # log["f1_train"].append(f1)

        # output
        #if epoch % 5 == 0:
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
                output, class_prediction = utils.constrain_prediction(output, class_prediction)

                loss = criterion(output, target.float())

                running_loss += loss.item()

                mean_accuracy, _ = accuracy(target, output, class_counts=train_dataloader.dataset.__label_counts__().to(DEVICE))
                running_accuracy += mean_accuracy.detach().cpu().numpy()

            running_loss /= (len(val_dataloader) * dataloader_val.batch_size)
            running_accuracy /= len(val_dataloader)
            # f1 = f1_score(y_true=target.detach().cpu().numpy(), y_pred=output.detach().cpu().numpy(),
                          # average='micro')

            scheduler.step()

            log["loss_val"].append(running_loss)
            log["accuracy_val"].append(running_accuracy)
            # log["f1_val"].append(f1)

            # output
            torch.save(model.state_dict(), "E:\\trained_models\\" + model_name + ".pt")
            with open("E:\\trained_models\\" + model_name + "_train_log.pkl", "wb") as f:
                pkl.dump(log, f)

            tqdm.write('Epoch {} (valid) -- loss: {:.4f} acc: {:.4f}'.format(epoch, running_loss,
                                                                             running_accuracy))

    torch.save(model.state_dict(), "E:\\trained_models\\" + model_name + ".pt")
    with open("E:\\trained_models\\" + model_name + "_train_log.pkl", "wb") as f:
        pkl.dump(log, f)


if __name__ == "__main__":
    print("device: ", DEVICE)

    dataset_train = dataset.NIHChestXRayDataset("E:\\NIH Chest Xray", mode="train")
    dataset_val = dataset.NIHChestXRayDataset("E:\\NIH Chest Xray", mode="val")

    dataloader_train = DataLoader(dataset_train, batch_size=24, shuffle=True, pin_memory=True, num_workers=8)
    dataloader_val = DataLoader(dataset_val, batch_size=24, shuffle=False, pin_memory=True, num_workers=8)

    label_counts = dataset_train.__label_counts__()
    weights = np.sqrt(1 / label_counts)
    weights /= weights.mean()

    model = ResNet18().to(DEVICE)
    lr = 0.00001
    criterion = nn.BCELoss(weight=weights.to(DEVICE, dtype=torch.float))
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    train(model, "nih_chest_resnet18", 35, dataloader_train, dataloader_val,
          criterion, optimizer, scheduler)

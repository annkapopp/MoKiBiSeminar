import torch
from torch.utils.data import Dataset
import cv2
import glob
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import pandas as pd


LABELS = ["Atelectasis", "Consolidation", "Pneumothorax", "Edema", "Effusion", "Pneumonia", "Cardiomegaly",
          "No Finding"]


def augmentation(image):
    image = TF.resize(image, [256, 256], antialias=True)
    image = transforms.RandomHorizontalFlip(0.5)(image)
    image = TF.equalize(image)
    image = TF.adjust_gamma(image, np.random.rand() + 0.5)
    angle = np.random.randint(-20, 20)
    image = TF.rotate(image, angle, TF.InterpolationMode.BILINEAR, expand=False)

    return image


class NIHChestXRayDataset(Dataset):
    def __init__(self, path_to_dir, mode: str):
        self.path_to_dir = path_to_dir
        self.mode = mode

        self.data = pd.read_pickle(path_to_dir + "/" + mode + "_data.pkl")

    def __getitem__(self, index):
        image_data = self.data.iloc[index]
        path = os.path.join(self.path_to_dir, "images", image_data["Image Index"])
        # print(path)
        image = torch.from_numpy(cv2.imread(path, cv2.IMREAD_GRAYSCALE)).unsqueeze(0)

        if self.mode == "train":
            # augmentation
            image = augmentation(image)

        image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))

        image_name = os.path.basename(path)
        one_hot = torch.from_numpy(np.asarray(self.data[self.data['Image Index'].str.contains(image_name)]
                                              ['target_vector'].values[0]).astype(int))
        label = LABELS[np.argmax(one_hot)]

        return image, label, one_hot, image_name

    def __len__(self):
        return len(self.data)

    def __label_counts__(self):
        label_counts = np.stack(self.data['one_hot'].values).sum(axis=0).astype(float)
        return torch.from_numpy(label_counts)


class CheXpertDataset(Dataset):
    def __init__(self, path_to_dir, mode: str):
        self.path_to_dir = path_to_dir
        self.mode = mode

        self.data = pd.read_pickle(path_to_dir + "/" + mode + "_data.pkl")

    def __getitem__(self, index):
        path = self.path_to_dir + "/" + self.data.iloc[index]["Path"]
        image = torch.from_numpy(cv2.imread(path, cv2.IMREAD_GRAYSCALE)).unsqueeze(0)

        if self.mode == "train":
            # augmentation
            image = augmentation(image)

        image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))

        image_name = os.path.basename(path)
        one_hot = torch.from_numpy(np.asarray(self.data[self.data['Path'].str.contains(image_name)]
                                              ['target_vector'].values[0]).astype(int))
        label = LABELS[np.argmax(one_hot)]

        return image, label, one_hot, image_name

    def __len__(self):
        return len(self.data)

    def __label_counts__(self):
        label_counts = np.stack(self.data['one_hot'].values).sum(axis=0).astype(float)
        return torch.from_numpy(label_counts)



if __name__ == "__main__":
    dataset_train = NIHChestXRayDataset(path_to_dir="E:\\NIH Chest Xray", mode="train")
    #dataset_test = NIHChestXRayDataset(path_to_dir="E:\\NIH Chest Xray", mode="test")
    #image = dataset_train.__getitem__(0)
    import multiprocessing as mp
    from torch.utils.data import DataLoader
    from time import time
    for num_workers in range(4, mp.cpu_count(), 2):
        train_loader = DataLoader(dataset_train, shuffle=True, num_workers=num_workers, batch_size=24, pin_memory=True)
        start = time()
        for i, data in enumerate(train_loader, 0):
            pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))


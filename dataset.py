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
            image = TF.resize(image, [256, 256], antialias=True)
            image = transforms.RandomHorizontalFlip(0.5)(image)
            image = TF.equalize(image)
            image = TF.adjust_gamma(image, np.random.rand() + 0.5)

        image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))

        image_name = os.path.basename(path)
        one_hot = torch.from_numpy(np.asarray(self.data[self.data['Image Index'].str.contains(image_name)]
                                              ['target_vector'].values[0]).astype(int))
        label = LABELS[np.argmax(one_hot)]

        return image, label, one_hot, image_name

    def __len__(self):
        return len(self.data)


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
            image = TF.resize(image, [512, 512], antialias=True)
            image = transforms.RandomHorizontalFlip(0.5)(image)
            image = TF.equalize(image)
            image = TF.adjust_gamma(image, np.random.rand() + 0.5)

        image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))

        image_name = os.path.basename(path)
        label = torch.from_numpy(self.data[self.data['Path'].str.contains(path)]
                                 ['target_vector'].values)

        return image, label, image_name

    def __len__(self):
        return len(self.data)



if __name__ == "__main__":
    dataset_train = NIHChestXRayDataset(path_to_dir="E:\\NIH Chest Xray", mode="train")
    dataset_test = NIHChestXRayDataset(path_to_dir="E:\\NIH Chest Xray", mode="test")
    image = dataset_train.__getitem__(0)


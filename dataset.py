import torch
from torch.utils.data import Dataset
import cv2
import glob
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import pandas as pd


LABELS_USED = ["Atelectasis", "Consolidation", "Pneumothorax", "Edema", "Effusion", "Pneumonia", "Cardiomegaly",
          "No Finding"]

ALL_LABELS = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
              'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia',
              'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum']


def augmentation(image):
    image = TF.resize(image, [256, 256], antialias=True)
    image = transforms.RandomHorizontalFlip(0.5)(image)
    image = TF.affine(image.unsqueeze(0),
                      translate=[np.random.rand()/10*image.shape[0], np.random.rand()/10*image.shape[1]],
                      angle=np.random.randint(-20, 20),
                      shear=[np.random.rand()/10*image.shape[0], np.random.rand()/10*image.shape[1]],
                      scale=1.0)
    image = TF.adjust_gamma(image, np.random.rand() + 0.5)

    return image


def normalise(image):
    # return ((image - torch.min(image)) / (torch.max(image) - torch.min(image)) * 2024) - 1024
    return ((image - 0) / (255 - 0) * 2024) - 1024


class NIHChestXRayDataset(Dataset):
    def __init__(self, path_to_dir, mode: str):
        self.path_to_dir = path_to_dir
        self.mode = mode

        self.data = pd.read_pickle(path_to_dir + "/" + mode + "_data.pkl")

    def __getitem__(self, index):
        image_data = self.data.iloc[index]
        path = os.path.join(self.path_to_dir, "images", image_data["Image Index"])
        image = torch.from_numpy(cv2.imread(path, cv2.IMREAD_GRAYSCALE)).unsqueeze(0)

        if self.mode == "train":
            # augmentation
            image = augmentation(image)
        else:
            image = TF.resize(image, [224, 224], antialias=True)

        image = normalise(image)

        image_name = os.path.basename(path)
        one_hot = torch.from_numpy(np.asarray(self.data[self.data['Image Index'].str.contains(image_name)]
                                              ['one hot'].values[0]).astype(int))
        label = LABELS_USED[np.argmax(one_hot)]

        return image, label, one_hot, image_name

    def __len__(self):
        return len(self.data)

    def __label_counts__(self):
        label_counts = np.stack(self.data['one hot'].values).sum(axis=0).astype(float)
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
        label = LABELS_USED[np.argmax(one_hot)]

        return image, label, one_hot, image_name

    def __len__(self):
        return len(self.data)

    def __label_counts__(self):
        label_counts = np.stack(self.data['one_hot'].values).sum(axis=0).astype(float)
        return torch.from_numpy(label_counts)



if __name__ == "__main__":
    dataset_train = NIHChestXRayDataset(path_to_dir="NIH Chest Xray", mode="train")
    dataset_val= NIHChestXRayDataset(path_to_dir="NIH Chest Xray", mode="val")
    dataset_test = NIHChestXRayDataset(path_to_dir="NIH Chest Xray", mode="test")
    #dataset_test = NIHChestXRayDataset(path_to_dir="E:\\NIH Chest Xray", mode="test")
    #image = dataset_train.__getitem__(0)

    print(dataset_train.__label_counts__(), dataset_train.__label_counts__() / dataset_train.__len__())
    print(dataset_val.__label_counts__(), dataset_val.__label_counts__() / dataset_val.__len__())
    print(dataset_test.__label_counts__(), dataset_test.__label_counts__() / dataset_test.__len__())


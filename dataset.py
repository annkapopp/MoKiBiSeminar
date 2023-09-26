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
    return ((image - torch.min(image)) / (torch.max(image) - torch.min(image)) * 2) - 1


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
            image = augmentation(image).squeeze(0)
        else:
            image = TF.resize(image, [256, 256], antialias=True)

        image = normalise(image)

        image_name = os.path.basename(path)
        target = torch.from_numpy(np.asarray(self.data[self.data['Image Index'].str.contains(image_name)]
                                             ['target vector'].values[0]).astype(int))
        label = LABELS_USED[np.argmax(target)]

        return image, label, target, image_name

    def __len__(self):
        return len(self.data)

    def __label_counts__(self):
        label_counts = np.stack(self.data['one hot'].values).sum(axis=0).astype(float)
        return torch.from_numpy(label_counts)

    def __target_label_counts__(self):
        label_counts = np.stack(self.data['target vector'].values).sum(axis=0).astype(float)
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
            image = augmentation(image).squeeze(0)
        else:
            image = TF.resize(image, [256, 256], antialias=True)

        image = normalise(image)


        image_name = self.data.iloc[index]["Path"]
        target_vector = torch.from_numpy(np.asarray(self.data[self.data['Path'].str.contains(image_name)]
                                                    ['target vector'].values[0]).astype(int))
        label = LABELS_USED[np.argmax(target_vector)]

        return image, label, target_vector, image_name

    def __len__(self):
        return len(self.data)

    def __label_counts__(self):
        label_counts = np.stack(self.data['one hot'].values).sum(axis=0).astype(float)
        return torch.from_numpy(label_counts)

    def __target_label_counts__(self):
        label_counts = np.stack(self.data['target vector'].values).sum(axis=0).astype(float)
        return torch.from_numpy(label_counts)



if __name__ == "__main__":
    #dataset_train = NIHChestXRayDataset(path_to_dir="NIH Chest Xray", mode="train")
    #dataset_val= NIHChestXRayDataset(path_to_dir="NIH Chest Xray", mode="val")
    #dataset_test = NIHChestXRayDataset(path_to_dir="NIH Chest Xray", mode="test")

    dataset_train = CheXpertDataset(path_to_dir="CheXpert", mode="train")
    dataset_val= CheXpertDataset(path_to_dir="CheXpert", mode="val")
    dataset_test = CheXpertDataset(path_to_dir="CheXpert", mode="test")

    print("Number of images: ", dataset_train.__len__() + dataset_val.__len__() + dataset_test.__len__())
    print("Number of patients: ", len(np.unique(np.vstack(dataset_train.data["Patient ID"].tolist() + dataset_val.data["Patient ID"].tolist()  +
                    dataset_test.data["Patient ID"].tolist()))))

    # Number of images per class and percentage
    print(dataset_train.__label_counts__(), dataset_train.__label_counts__() / dataset_train.__len__())
    print(dataset_val.__label_counts__(), dataset_val.__label_counts__() / dataset_val.__len__())
    print(dataset_test.__label_counts__(), dataset_test.__label_counts__() / dataset_test.__len__())



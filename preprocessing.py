import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import dataset

LABELS = dataset.LABELS


def preprocess_NIHChest(path_to_dir):
    all_labels = ["Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema", "Emphysema",
                       "Fibrosis", "Effusion", "Pneumonia", "Pleural_thickening", "Cardiomegaly", "Nodule Mass",
                       "Hernia", "No findings"]

    data = pd.read_csv(path_to_dir + "/" + "Data_Entry_2017.csv",
                            usecols=["Image Index", "Finding Labels", "Patient ID"], index_col=None)

    folder_num = np.zeros(len(data))
    folder_num[0:4999] = 1
    folder_num[5000:15000] = 2
    folder_num[15000:25000] = 3
    folder_num[25000:35000] = 4
    folder_num[35000:45000] = 5
    folder_num[45000:55000] = 6
    folder_num[55000:65000] = 7
    folder_num[65000:75000] = 8
    folder_num[75000:85000] = 9
    folder_num[85000:95000] = 10
    folder_num[95000:105000] = 11
    folder_num[105000:] = 12
    data['folder_num'] = list(folder_num.astype(int))

    data = data[data["Finding Labels"].isin(LABELS)].reset_index()

    for label in LABELS:
        data[label] = data["Finding Labels"].map(
            lambda result: 1 if label in result else 0)

    data['target_vector'] = data.apply(lambda target: [target[LABELS].values], 1).map(
        lambda target: target[0])

    # print(len(pd.unique(data['Patient ID'])))

    # data.to_pickle(path_to_dir + "/data.pkl")

    gs = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=0)
    train_idx, test_idx = next(gs.split(data['Image Index'], data['target_vector'], groups=data['Patient ID']))

    test_df = data.iloc[test_idx]
    train_df = data.iloc[train_idx]

    gs = GroupShuffleSplit(n_splits=1, test_size=.1, random_state=0)
    train_idx, val_idx = next(gs.split(train_df['Image Index'], train_df['target_vector'], groups=train_df['Patient ID']))
    val_df = data.iloc[val_idx]
    train_df = data.iloc[train_idx]

    train_df.to_pickle(path_to_dir + "/train_data.pkl")
    val_df.to_pickle(path_to_dir + "/val_data.pkl")
    test_df.to_pickle(path_to_dir + "/test_data.pkl")


def preprocess_CheXpert(path_to_dir):
    all_labels = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
                    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
                    "Pleural Other", "Fracture", "Support Devices"]

    data_train = pd.read_csv(path_to_dir + "/" + "train.csv", usecols=["Path"] + all_labels, index_col=None)
    data_val = pd.read_csv(path_to_dir + "/" + "valid.csv", usecols=["Path"] + all_labels, index_col=None)
    data = pd.concat([data_train, data_val]).fillna(0)
    data = data.apply(lambda x: x.replace(
        {"CheXpert-v1.0-small/": ""}, regex=True
    ))
    data = data.rename(columns={"No Finding": "No findings", "Pleural Effusion": "Effusion"})

    data = data[data.columns[data.columns.isin(["Path"] + LABELS)]]
    data = data.replace([-1], 0)

    data['target_vector'] = data.apply(lambda target: [target[LABELS].values.astype(int)], 1).map(
        lambda target: target[0])

    data['num_labels'] = data.sum(axis=1, numeric_only=True)
    data = data.loc[data['num_labels'] != 0]
    data = data.drop("num_labels", axis=1)

    data["Patient ID"] = [row.split("/")[1].split("patient")[-1] for row in data["Path"].values]
    # print(len(pd.unique(data['Patient ID'])))

    gs = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=0)
    train_idx, test_idx = next(gs.split(data['Path'], data['target_vector'], groups=data['Patient ID']))
    test_df = data.iloc[test_idx]
    train_df = data.iloc[train_idx]

    gs = GroupShuffleSplit(n_splits=1, test_size=.1, random_state=0)
    train_idx, val_idx = next(gs.split(train_df['Path'], train_df['target_vector'], groups=train_df['Patient ID']))
    val_df = data.iloc[val_idx]
    train_df = data.iloc[train_idx]

    train_df.to_pickle(path_to_dir + "/train_data.pkl")
    val_df.to_pickle(path_to_dir + "/val_data.pkl")
    test_df.to_pickle(path_to_dir + "/test_data.pkl")



if __name__ == "__main__":
    preprocess_NIHChest(path_to_dir="/Volumes/Temp/NIH Chest Xray")
    preprocess_CheXpert(path_to_dir="/Volumes/Temp/CheXpert")

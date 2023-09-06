import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import dataset

LABELS = dataset.LABELS


def preprocess_NIHChest(path_to_dir):
    all_labels = ["Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema", "Emphysema",
                       "Fibrosis", "Effusion", "Pneumonia", "Pleural_thickening", "Cardiomegaly", "Nodule Mass",
                       "Hernia", "No Finding"]

    data = pd.read_csv(path_to_dir + "/" + "Data_Entry_2017.csv",
                            usecols=["Image Index", "Finding Labels", "Patient ID"], index_col=None)

    data = data[data["Finding Labels"].isin(LABELS)].reset_index()

    for label in LABELS:
        data[label] = data["Finding Labels"].map(
            lambda result: 1 if label in result else 0)

    data['one_hot'] = data.apply(lambda target: [target[LABELS].values], 1).map(
        lambda target: target[0])

    target_vectors = np.stack(data['one_hot'].values)
    data['target_vector'] = target_vectors.tolist()

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


def preprocess_CheXpert(path_to_dir, crop_size):
    all_labels = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
                    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
                    "Pleural Other", "Fracture", "Support Devices"]

    data_train = pd.read_csv(path_to_dir + "/" + "train.csv", usecols=["Path"] + all_labels, index_col=None)
    data_val = pd.read_csv(path_to_dir + "/" + "valid.csv", usecols=["Path"] + all_labels, index_col=None)
    data = pd.concat([data_train, data_val]).fillna(0)
    data = data.apply(lambda x: x.replace(
        {"CheXpert-v1.0-small/": ""}, regex=True
    ))
    data = data.rename(columns={"Pleural Effusion": "Effusion"})

    data = data[data.columns[data.columns.isin(["Path"] + LABELS)]]
    data = data.replace([-1], 0)
   

    data = data[['Path', 'Atelectasis', 'Consolidation', 'Pneumothorax', 'Edema', 'Effusion',
                 'Pneumonia', 'Cardiomegaly', 'No Finding']]

    data['one_hot'] = data.apply(lambda target: [target[LABELS].values.astype(int)], 1).map(
        lambda target: target[0])

    data['num_labels'] = data.sum(axis=1, numeric_only=True)
    data = data.loc[data['num_labels'] != 0]
    data = data.drop("num_labels", axis=1)

    data["Patient ID"] = [row.split("/")[1].split("patient")[-1] for row in data["Path"].values]

    data = data.reset_index(drop=True)
    # print(len(pd.unique(data['Patient ID'])))

    label_percentage = np.stack(data['one_hot'].values).sum(axis=0).astype(float) / len(data)
    new_data = pd.DataFrame()
    for label, p in zip(LABELS, label_percentage):
        label_data = data[label]
        n_new = int(crop_size * p)
        print(p, n_new)
        indices = label_data.index.values[:n_new]
        new_data = new_data._append(data.iloc[indices])

    data = new_data
    print(len(data))

    target_vectors = np.stack(data['one_hot'].values)
    data['target_vector'] = target_vectors.tolist()

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
    #preprocess_NIHChest(path_to_dir="E:\\NIH Chest Xray")
    preprocess_CheXpert(path_to_dir="E:\\CheXpert", crop_size=112120)

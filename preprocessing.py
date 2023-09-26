import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
import dataset

LABELS_USED = dataset.LABELS_USED
ALL_LABELS = dataset.ALL_LABELS


def preprocess_NIHChest(path_to_dir):
    all_nih_labels = ["Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema", "Emphysema",
                  "Fibrosis", "Effusion", "Pneumonia", "Pleural_thickening", "Cardiomegaly", "Nodule Mass",
                  "Hernia", "No Finding"]

    # load only relevant columns of csv
    data = pd.read_csv(path_to_dir + "/" + "Data_Entry_2017.csv",
                       usecols=["Image Index", "Finding Labels", "Patient ID"], index_col=None)

    # create column encoding if image contains class
    for label in LABELS_USED:
        data[label] = data["Finding Labels"].map(
            lambda result: 1 if label in result else 0)

    # columns to one hot vector
    data['one hot'] = data.apply(lambda target: [target[LABELS_USED].values], 1).map(
        lambda target: target[0])

    # "no findings" not included in target
    one_hot_vectors = np.vstack(data['one hot'].values)
    data['target vector'] = one_hot_vectors[:, :-1].tolist()

    # exclude images with no labels
    counts_per_label = one_hot_vectors.sum(axis=0).astype(float)
    counts_per_image = one_hot_vectors.sum(axis=1).astype(float)
    label_idx = np.where(counts_per_image != 0)[0]
    data = data.iloc[label_idx].reset_index(drop=True)

    # adjust number of "no finding" for less unbalanced data
    one_hot_vectors = np.vstack(data['one hot'].values)
    no_findings_idx = np.where(one_hot_vectors[:, -1] == 1)[0]
    n_no_findings_new = int(np.median(counts_per_label[:-1]))
    data = data.drop(no_findings_idx[n_no_findings_new:])

    # only keep entries with overlapping labels to chexpert
    data = data[data["Finding Labels"].str.contains('|'.join(LABELS_USED))].reset_index(drop=True)

    # encode one hot vectors as integers
    class_values = np.argmax(np.stack(data['one hot'].values), axis=1)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for train_i, test_i in sgkf.split(data['Image Index'], class_values, groups=data["Patient ID"]):
        train_idx = train_i
        test_idx = test_i
        break

    test_idx = np.random.permutation(test_idx)
    val_idx = test_idx[: len(test_idx) // 2]
    test_idx = test_idx[len(test_idx) // 2:]

    test_df = data.iloc[test_idx]
    val_df = data.iloc[val_idx]
    train_df = data.iloc[train_idx]

    train_df.to_pickle(path_to_dir + "/train_data.pkl")
    val_df.to_pickle(path_to_dir + "/val_data.pkl")
    test_df.to_pickle(path_to_dir + "/test_data.pkl")
    

def preprocess_CheXpert(path_to_dir):
    all_labels = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
                  "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
                  "Pleural Other", "Fracture", "Support Devices"]

    data_train = pd.read_csv(path_to_dir + "/" + "train.csv", usecols=["Path", "Frontal/Lateral"] + all_labels,
                             index_col=None)
    data_val = pd.read_csv(path_to_dir + "/" + "valid.csv", usecols=["Path", "Frontal/Lateral"] + all_labels,
                           index_col=None)

    # fill nan with 0
    data = pd.concat([data_train, data_val]).fillna(0)

    # remove file path from image name
    data = data.apply(lambda x: x.replace(
        {"CheXpert-v1.0-small/": ""}, regex=True
    ))

    # adjust labels
    data = data.rename(columns={"Pleural Effusion": "Effusion"})

    # only keep frontal view and remove lateral views
    data = data[data["Frontal/Lateral"] == "Frontal"].reset_index()

    # only keep columns that are in LABELS_USED
    data = data[data.columns[data.columns.isin(["Path"] + LABELS_USED)]]

    # replace all unsure entries
    data = data.replace([-1], 0)

    # sort columns
    data = data[['Path', 'Atelectasis', 'Consolidation', 'Pneumothorax', 'Edema', 'Effusion',
                 'Pneumonia', 'Cardiomegaly', 'No Finding']]

    # encode labels as one hot vectors
    data['one hot'] = data.apply(lambda target: [target[LABELS_USED].values.astype(int)], 1).map(
        lambda target: target[0])

    # only keep images which have annotations
    one_hot_vectors = np.vstack(data['one hot'].values)
    counts_per_label = one_hot_vectors.sum(axis=0).astype(float)
    counts_per_image = one_hot_vectors.sum(axis=1).astype(float)
    label_idx = np.where(counts_per_image != 0)[0]
    data = data.iloc[label_idx].reset_index(drop=True)

    # assign patient ids
    data["Patient ID"] = [row.split("/")[1].split("patient")[-1] for row in data["Path"].values]

    data = data.reset_index(drop=True)

    # delete "no finding" column -> zero vector
    one_hot_vectors = np.vstack(data['one hot'].values)
    one_hot_vectors = one_hot_vectors[:, :-1]
    data['target vector'] = one_hot_vectors.tolist()

    # encode one hot vectors as integers
    class_values = np.argmax(np.vstack(data['one hot'].values), axis=1)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for train_i, test_i in sgkf.split(data['Path'], class_values, groups=data["Patient ID"]):
        train_idx = train_i
        test_idx = test_i
        break

    test_idx = np.random.permutation(test_idx)
    val_idx = test_idx[: len(test_idx) // 2]
    test_idx = test_idx[len(test_idx) // 2:]

    test_df = data.iloc[test_idx]
    val_df = data.iloc[val_idx]
    train_df = data.iloc[train_idx]

    train_df.to_pickle(path_to_dir + "/train_data.pkl")
    val_df.to_pickle(path_to_dir + "/val_data.pkl")
    test_df.to_pickle(path_to_dir + "/test_data.pkl")


if __name__ == "__main__":
    # preprocess_NIHChest(path_to_dir="NIH Chest Xray")
    preprocess_CheXpert(path_to_dir="CheXpert")

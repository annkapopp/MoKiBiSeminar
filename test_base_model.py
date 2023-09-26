import torch
from torch.utils.data import DataLoader
import pickle as pkl
from sklearn.metrics import multilabel_confusion_matrix

from base_models import DenseNet121
import dataset
import utils
from tqdm import tqdm

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test(model, model_name, dataloader):
    log = {
        'image name': [],
        'target': [],
        'prediction': [],
    }
    with torch.no_grad():
        model.eval()

        for input, _, target, image_name in tqdm(dataloader, position=0, leave=True):
            input = input.to(DEVICE)
            target = target.to(DEVICE)
            output = model(input)

            class_prediction = utils.get_class_prediction(output)
            _, class_prediction = utils.constrain_prediction(output, class_prediction)
            class_prediction = class_prediction.detach().cpu().numpy()

            log["target"].append(target.detach().cpu().numpy())
            log["prediction"].append(class_prediction)

            target = target.detach().cpu().numpy()

            log["image name"].append(image_name)

        with open("results/" + model_name + "_nih_test.pkl", "wb") as f:
            pkl.dump(log, f)


if __name__ == "__main__":
    print("device: ", DEVICE)

    model_name = "chex_densenet"
    dataset_test = dataset.NIHChestXRayDataset("NIH Chest Xray", mode="test")
    # dataset_test = dataset.CheXpertDataset("CheXpert", mode="test")

    print(dataset_test.__target_label_counts__())

    dataloader = DataLoader(dataset_test, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)
    model = DenseNet121(pretrained=False)
    model.load_state_dict(torch.load("trained_models/" + model_name + ".pt", map_location=torch.device('cpu')))

    test(model.to(DEVICE), model_name, dataloader)

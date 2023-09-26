"""
This code has been largely adopted from https://github.com/qinenergy/cotta.
"""


import torch
import logging
from torch.utils.data import DataLoader
import pickle as pkl
from tqdm import tqdm

from base_models import DenseNet121
import cotta
import dataset
import utils

logger = logging.getLogger(__name__)
DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


def evaluate(model_name, dataset):
    log = {
        'batch num': [],
        'image name': [],
        "prediction": [],
        "target": [],
        "loss": []
    }

    model = DenseNet121().to(DEVICE)
    model.load_state_dict(torch.load("trained_models/" + model_name + ".pt"))

    cotta_model = setup_cotta(model).to(DEVICE)
    logger.info(f"model for evaluation: %s", model_name)

    batch_size = 2
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False,
                            drop_last=True)

    batch_num = 0

    for input, _, target, image_name in tqdm(dataloader, position=0, leave=True):
        input = input.to(DEVICE)
        output = cotta_model(input)

        class_prediction = utils.get_class_prediction(output)
        _, class_prediction = utils.constrain_prediction(output, class_prediction)
        class_prediction = class_prediction.detach().cpu().numpy()

        log['batch num'].extend([batch_num] * batch_size)
        log["image name"].extend(image_name)
        log["prediction"].extend(list(class_prediction))
        log["target"].extend(list(target.detach().cpu().numpy()))
        log["loss"].extend([cotta_model.loss_student] * batch_size)

        batch_num += 1

    with open("trained_models/" + model_name + "_cotta_batch2.pkl", "wb") as f:
        pkl.dump(log, f)


def setup_cotta(model):
    model = cotta.configure_model(model)

    params, param_names = cotta.collect_params(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99), weight_decay=0.0)
    cotta_model = cotta.CoTTA(model, optimizer,
                              steps=1,
                              episodic=False,
                              mt_alpha=0.999,
                              rst_m=0.01,
                              ap=0.92)

    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)

    return cotta_model


if __name__ == "__main__":
    model_name = "chex_densenet"
    dataset = dataset.NIHChestXRayDataset("NIH Chest Xray", mode="test")
    # dataset = dataset.CheXpertDataset("CheXpert", mode="test")
    evaluate(model_name, dataset)

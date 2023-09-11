import torch
from torch import nn
import torchvision
import torchxrayvision as txrv
import torch.nn.functional as F


class ResNet18(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        if pretrained:
            weights = torchvision.models.ResNet18_Weights
            self.model = torchvision.models.resnet18(weights.IMAGENET1K_V1)
        else:
            self.model = torchvision.models.resnet18()
        self.model.fc = nn.Linear(in_features=512, out_features=8, bias=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)


class DenseNet121old(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        if pretrained:
            weights = torchvision.models.DenseNet121_Weights
            self.model = torchvision.models.densenet121(weights.IMAGENET1K_V1)
        else:
            self.model = torchvision.models.densenet121()
        self.model.classifier = nn.Linear(in_features=1024, out_features=8, bias=True)
        self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)


class DenseNet121(nn.Module):
    def __init__(self, weights: str):
        super().__init__()
        if weights == "nih":
            model = txrv.models.DenseNet(weights="densenet121-res224-nih", apply_sigmoid=False)
        elif weights == "chex":
            model = txrv.models.DenseNet(weights="densenet121-res224-chex", apply_sigmoid=False)

        self.features = model.features
        self.classifier = model.classifier
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    model_nih = DenseNet121(weights="nih")
    model_chex = DenseNet121(weights="chex")
    model = DenseNet121old()

    x = torch.ones((1, 1, 224, 224))
    y = model_nih(x)


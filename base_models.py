from torch import nn
import torchvision


class ResNet18(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        if pretrained:
            weights = torchvision.models.ResNet18_Weights
            self.model = torchvision.models.resnet18(weights.IMAGENET1K_V1)
        else:
            self.model = torchvision.models.resnet18()
        self.model.fc = nn.Linear(in_features=512, out_features=7, bias=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)


class DenseNet121(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        if pretrained:
            weights = torchvision.models.DenseNet121_Weights
            self.model = torchvision.models.densenet121(weights.IMAGENET1K_V1)
        else:
            self.model = torchvision.models.densenet121()
        self.model.classifier = nn.Linear(in_features=1024, out_features=7, bias=True)
        self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)


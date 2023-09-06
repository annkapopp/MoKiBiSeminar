from torch import nn
import torchvision


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights
        self.model = torchvision.models.resnet18(weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(in_features=512, out_features=8, bias=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)


if __name__ == "__main__":
    model = ResNet18()

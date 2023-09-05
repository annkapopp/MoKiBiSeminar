from torch import nn
import torchvision


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18()
        self.model.fc = nn.Linear(in_features=512, out_features=8, bias=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = ResNet18()

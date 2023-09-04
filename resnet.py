from torch import nn
import torchvision


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18()
        self.model.fc = nn.Linear(in_features=512, out_features=8, bias=True)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = ResNet18()

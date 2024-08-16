import torch
import torch.nn as nn

class PreactivatedBasicResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride = 1):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_dim)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, stride = stride, padding=1)

        self.bn2 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, stride = 1, padding=1)

        self.skip_connection = nn.Sequential()
        if stride != 1 or in_dim != out_dim:
            self.skip_connection = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1, stride=stride),
                                                 nn.BatchNorm2d(out_dim))
            
    def forward(self, x):

        residue = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += self.skip_connection(residue)

        return out


class Resnet110(nn.Module):
    def __init__(self, num_classes = 1000):
        super().__init__()
        
        # initial layers
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3,2,1)

        # Residual layers
        self.layer1 = self._make_layer(64, 64, 18,stride=1)
        self.layer2 = self._make_layer(64, 128, 18, stride=2)
        self.layer3 = self._make_layer(128, 256, 18, stride=2)
        self.layer4 = self._make_layer(256, 512, 18, stride=2)

        # final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)


    def _make_layer(self, in_dim, out_dim, blocks, stride):
        layers = []

        layers.append(PreactivatedBasicResBlock(in_dim, out_dim, stride))
        for _ in range(1, blocks):
            layers.append(PreactivatedBasicResBlock(out_dim, out_dim, stride=1))

        return nn.Sequential(*layers)


    def forward(self , x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


model = Resnet110(num_classes=1000)
print(model)
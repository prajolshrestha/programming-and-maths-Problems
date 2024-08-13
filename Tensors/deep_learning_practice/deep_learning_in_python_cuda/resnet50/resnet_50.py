import torch 
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, downsample = None):
        super().__init__()

        self.expansion = 4

        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_dim)

        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.conv3 = nn.Conv2d(out_dim, out_dim * self.expansion, kernel_size=1, stride = 1)
        self.bn3 = nn.BatchNorm2d(out_dim * self.expansion)

        self.downsample = downsample

    def forward(self, x):
        identity = x 

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out


class Resnet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.in_dim = 64

        # Initial layer
        # (3, 224, 224) ==> (64, 112, 112)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # (64, 112, 112) ==> (64, 56, 56)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Bottleneck layers
        # (64, 56, 56) ==> (256, 56, 56)
        self.layer1 = self._make_layer(64,256,3, stride = 1)
        # (256, 56, 56) ==> (512, 28, 28)
        self.layer2 = self._make_layer(256,512,4, stride = 2)
        # (512, 28, 28) ==> (1024, 14, 14)
        self.layer3 = self._make_layer(512,1024,6, stride = 2)
        # (1024, 14, 14) ==> (2048, 7, 7)
        self.layer4 = self._make_layer(1024,2048,3, stride = 2)

        # Final layer
        # (2048, 7, 7) ==> (2048, 1 ,1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # (2048, 1, 1) ==> (1000, 1, 1)
        self.fc = nn.Linear(2048 * 4, num_classes)


    def _make_layer(self, in_dim, out_dim, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_dim != out_dim * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_dim, out_dim * 4, kernel_size=1, stride = stride),
                nn.BatchNorm2d(out_dim * 4)     
            )
        
        layers = []
        layers.append(Bottleneck(in_dim, out_dim, stride, downsample))
        
        self.in_dim = out_dim * 4

        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_dim, out_dim))
        
        return nn.Sequential(*layers)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


model = Resnet50(num_classes=1000)
print(model)
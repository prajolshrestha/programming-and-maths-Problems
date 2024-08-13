import torch 
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride = 1):
        super().__init__()

        # conv with flexible stride parameter (it allows to optionally downsample the spatial dimensions of the input)
        # Stride > 1: reduces the spatial dimensions (height and width) of the feature maps
        # This is crucial for gradually reducing the spatial size of the feature maps as we go deeper into the network,
        # which helps in building a hierarchy of features and reducing computational cost.
        # early downsampling ==> leads to conv2 operate on smaller feature maps, which is more computationally efficient
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1) # controlled downsampling
        self.bn1 = nn.BatchNorm2d(out_dim)
        
        self.relu = nn.ReLU()

        # conv with stride = 1: Preserve the spatial dimensions
        # Purpose: Increase the networks depth and complexity without further changing the spatial dimensions
        self.conv2 = nn.Conv2d(out_dim,out_dim, kernel_size=3,stride=1,padding=1) # Maintain info flow: no further spatial information is lost
        self.bn2 = nn.BatchNorm2d(out_dim)

        # skip connection
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_dim != out_dim:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_dim)
            )

    
    def forward(self, x):
        residue = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.skip_connection(residue)
        out = self.relu(out)

        return out


class Resnet18(nn.Module):
    def __init__(self, num_classes = 1000):
        super().__init__()

        # Initial layers
        # (3 x 224 x 224) ==> (64 x 112 x 112)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # (64 x 112 x 112) ==> (64 x 112 x 112)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True) # inplace ==> to optimize memory usage # modifies input tensor directly rather than creating a new tensor for the output
        # (64 x 112 x 112) ==> (64 x 56 x 56)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layers
        # (64 x 56 x 56) ==> (64 x 56 x 56)
        self.layer1 = self._make_layers(64,64,stride=1) # stride = 1 ==> to capture low level features like edge and textures, it requires high spatial resolution
        # (64 x 56 x 56) ==> (128 x 28 x 28)
        self.layer2 = self._make_layers(64,128,stride=2) # stride > 2 ==> increase receptive field to see abstract and high level features
        # (128 x 28 x 28) ==> (256 x 14 x 14)
        self.layer3 = self._make_layers(128,256,stride=2)
        # (256 x 14 x 14) ==> (512 x 7 x 7)
        self.layer4 = self._make_layers(256,512,stride=2)

        # Final layers
        # (512 x 7 x 7) ==> (512 x 1 x 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # (512 x 1 x 1) ==> 1000
        self.fc = nn.Linear(512, num_classes)



    def _make_layers(self, in_dim, out_dim, stride):

        layers = []
        layers.append(ResidualBlock(in_dim, out_dim, stride))
        layers.append(ResidualBlock(out_dim, out_dim))

        return nn.Sequential(*layers)
    
    def forward(self, x):

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


model = Resnet18(num_classes=1000)
print(model)





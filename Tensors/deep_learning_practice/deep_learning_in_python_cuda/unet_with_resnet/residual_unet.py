import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.identity = nn.Sequential()
        if stride != 1 or in_dim != out_dim:
            self.identity = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size= 1, stride=stride),
                                          nn.BatchNorm2d(out_dim))
            
    def forward(self, x):
        residue = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.identity(residue)
        out = self.relu(out)

        return out



class ResidualUnet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.enc1 = self.res_conv_block(in_dim, 64)
        self.enc2 = self.res_conv_block(64, 128)
        self.enc3 = self.res_conv_block(128, 256)
        self.enc4 = self.res_conv_block(256, 512)

        self.bottleneck = self.res_conv_block(512, 1024)

        self.dec1 = self.res_upconv_block(1024, 512)
        self.dec2 = self.res_upconv_block(512, 256)
        self.dec3 = self.res_upconv_block(256, 128)
        self.dec4 = self.res_upconv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_dim, 1, 1, 0)

    def res_conv_block(self, in_dim, out_dim):
        return nn.Sequential(ResidualBlock(in_dim, out_dim, stride=1))

    def res_upconv_block(self, in_dim, out_dim):
        return nn.Sequential(nn.ConvTranspose2d(in_dim, out_dim, 2, 2, 0),
                             ResidualBlock(out_dim, out_dim, stride=1))

    def forward(self, x):
        
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))

        b = self.bottleneck(nn.MaxPool2d(2)(e4))

        # Decoder
        d1 = self.dec1(torch.cat([e4, nn.Upsample(scale_factor=2)(b)],dim=1))
        d2 = self.dec2(torch.cat([e3, nn.Upsample(scale_factor=2)(d1)],dim=1))
        d3 = self.dec3(torch.cat([e2, nn.Upsample(scale_factor=2)(d2)],dim=1))
        d4 = self.dec4(torch.cat([e1, nn.Upsample(scale_factor=2)(d3)],dim=1))

        out = self.final_conv(d4)

        return out


model = ResidualUnet(3, 2)

print(model)

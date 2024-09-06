import torch
import torch.nn as nn

class unet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.enc1 = self.conv_block(in_dim, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.bottleneck = self.conv_block(512, 1024)

        self.dec1 = self.upconv_block(1024, 512)
        self.dec2 = self.upconv_block(512, 256)
        self.dec3 = self.upconv_block(256, 128)
        self.dec4 = self.upconv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_dim, 1, 1, 0)



    def conv_block(self, in_dim, out_dim):
        return nn.Sequential(nn.Conv2d(in_dim, out_dim, 3, 1, 1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(out_dim, out_dim, 3, 1, 1),
                             nn.ReLU(inplace=True)
                             )
    
    def upconv_block(self, in_dim, out_dim):
        return nn.Sequential(nn.ConvTranspose2d(in_dim, out_dim, 2,2,0),
                             self.conv_block(out_dim, out_dim)
                             )
    
    def forward(self, x):
        """
        Performs the forward pass of the U-Net model.
        
        This method implements the encoder and decoder stages of the U-Net model. The encoder stage
        consists of a series of convolutional and max pooling layers that progressively downsample
        the input image. The decoder stage consists of a series of transposed convolutional and
        concatenation layers that progressively upsample the feature maps and combine them with
        the corresponding feature maps from the encoder stage.
        
        Args:
            x (torch.Tensor): The input image tensor of shape (batch_size, 3, height, width).
        
        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_classes, height, width).
        """
        # Encoder
        # (bs, 3, 224, 224) --> (bs, 64, 224,224)
        e1 = self.enc1(x)
        # (bs, 64, 224, 224) --> (bs, 128, 112, 112)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        # (bs, 128, 112, 112) --> (bs, 256, 56, 56)
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        # (bs, 256, 56, 56) --> (bs, 512, 28, 28)
        e4 = self.enc4(nn.MaxPool2d(2)(e3))

        # (bs, 512, 28, 28) --> (bs, 1024, 14, 14)
        b = self.bottleneck(nn.MaxPool2d(2)(e4))

        # Decoder
        # (bs, 1024, 14, 14) --> (bs, 512, 28, 28)
        d1 = self.dec1(torch.cat([e4, nn.Upsample(scale_factor=2)(b)],dim=1))
        # (bs, 512, 28, 28) --> (bs, 256, 56, 56) 
        d2 = self.dec2(torch.cat([e3, nn.Upsample(scale_factor=2)(d1)],dim=1))
        # (bs, 256, 56, 56) --> (bs, 128, 112, 112)
        d3 = self.dec3(torch.cat([e2, nn.Upsample(scale_factor=2)(d2)],dim=1))
        # (bs, 128, 112, 112) --> (bs, 64, 224, 224)
        d4 = self.dec4(torch.cat([e1, nn.Upsample(scale_factor=2)(d3)],dim=1))

        # (bs, 64, 224, 224) --> (bs, 2, 224, 224)
        return self.final_conv(d4)


model = unet(3,2)
print(model)


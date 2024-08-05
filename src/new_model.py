import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, batch_norm=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if batch_norm:
            layers.insert(1, nn.BatchNorm2d(mid_channels))
            layers.insert(-1, nn.BatchNorm2d(out_channels))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class UNet3Plus(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, upscale_mode="bilinear"):
        super(UNet3Plus, self).__init__()
        
        self.upscale_mode = upscale_mode
        
        # VGG16 encoder
        vgg16 = models.vgg16_bn(pretrained=True)
        self.encoder1 = nn.Sequential(*vgg16.features[:6])  # 64
        self.encoder2 = nn.Sequential(*vgg16.features[6:13])  # 128
        self.encoder3 = nn.Sequential(*vgg16.features[13:23])  # 256
        self.encoder4 = nn.Sequential(*vgg16.features[23:33])  # 512
        self.encoder5 = nn.Sequential(*vgg16.features[33:43])  # 512

        # Modify the first convolution layer to accept 1 channel input instead of 3
        self.encoder1[0] = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2)
        
        # Decoder
        self.decoder5 = nn.Conv2d(512, 512, kernel_size=1)
        self.decoder4 = nn.Conv2d(512, 256, kernel_size=1)
        self.decoder3 = nn.Conv2d(256, 128, kernel_size=1)
        self.decoder2 = nn.Conv2d(128, 64, kernel_size=1)
        
        # Final Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        
        dec5 = F.interpolate(self.decoder5(enc5), scale_factor=2, mode=self.upscale_mode)
        dec4 = F.interpolate(self.decoder4(dec5 + enc4), scale_factor=2, mode=self.upscale_mode)
        dec3 = F.interpolate(self.decoder3(dec4 + enc3), scale_factor=2, mode=self.upscale_mode)
        dec2 = F.interpolate(self.decoder2(dec3 + enc2), scale_factor=2, mode=self.upscale_mode)
        
        return self.final_conv(dec2 + enc1)

# 모델 테스트 코드
if __name__ == "__main__":
    model = UNet3Plus(in_channels=1, out_channels=2, batch_norm=True, upscale_mode="bilinear")
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print(y.shape)

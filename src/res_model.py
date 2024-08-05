import torch
import torchvision

class Block(torch.nn.Module):
    def __init__(self, in_channels, mid_channel, out_channels, batch_norm=False):
        super().__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=mid_channel, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, padding=1)
        
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = torch.nn.BatchNorm2d(mid_channel)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)
            
    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = torch.nn.functional.relu(x, inplace=True)
        
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        out = torch.nn.functional.relu(x, inplace=True)
        return out

class ResNetUNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False, upscale_mode="nearest"):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.upscale_mode = upscale_mode

        # Initialize ResNet152
        resnet152 = torchvision.models.resnet152(pretrained=True)
        self.encoder = torch.nn.ModuleList([
            resnet152.conv1,  # 64 channels
            resnet152.bn1,
            resnet152.relu,
            resnet152.maxpool,
            resnet152.layer1,  # 256 channels
            resnet152.layer2,  # 512 channels
            resnet152.layer3,  # 1024 channels
            resnet152.layer4   # 2048 channels
        ])
        
        # Define the initial conv to convert 1 channel input to 3 channels
        self.init_conv = torch.nn.Conv2d(in_channels, 3, kernel_size=1)

        # Define the bottleneck layer
        self.center = Block(2048, 2048, 2048, batch_norm)  # Bottleneck layer with 2048 channels

        # Define the decoder blocks
        self.dec5 = Block(2048 + 2048, 1024, 512, batch_norm)  # 2048 from center + 2048 from enc5
        self.dec4 = Block(512 + 1024, 512, 256, batch_norm)    # 512 from dec5 + 1024 from enc4
        self.dec3 = Block(256 + 512, 256, 128, batch_norm)        # 256 from dec4 + 512 from enc3
        self.dec2 = Block(128 + 256, 128, 64, batch_norm)        # 128 from dec3 + 256 from enc2
        self.dec1 = Block(64 + 64, 64, 32, batch_norm)         # 64 from dec2 + 64 from enc1

        # Supplementary layers
        self.sup5 = Block(2048, 2048, 2048, batch_norm)
        self.sup4 = Block(1024, 1024, 1024, batch_norm)
        self.sup3 = Block(512, 512, 512, batch_norm)
        self.sup2 = Block(256, 256, 256, batch_norm)
        self.sup1 = Block(64, 64, 64, batch_norm)

        # Define the final output layer
        self.out = torch.nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1)

    def up(self, x, target_size):
        return torch.nn.functional.interpolate(x, size=target_size, mode=self.upscale_mode)

    def down(self, x):
        return torch.nn.functional.max_pool2d(x, kernel_size=2)

    def forward(self, x):
        init_conv = torch.nn.functional.relu(self.init_conv(x), inplace=True)

        # Extract features from ResNet152
        enc1 = self.encoder[0](init_conv)  # 64 channels
        enc1 = self.encoder[1](enc1)
        enc1 = self.encoder[2](enc1)
        enc1 = self.encoder[3](enc1)
        enc2 = self.encoder[4](enc1)  # 256 channels
        enc3 = self.encoder[5](self.down(enc2))  # 512 channels
        enc4 = self.encoder[6](self.down(enc3))  # 1024 channels
        enc5 = self.encoder[7](self.down(enc4))  # 2048 channels

        # Supplementary layers in encoder
        enc5 = self.sup5(enc5)  # 2048 channels
        enc4 = self.sup4(enc4)  # 1024 channels
        enc3 = self.sup3(enc3)  # 512 channels
        enc2 = self.sup2(enc2)  # 256 channels
        enc1 = self.sup1(enc1)  # 64 channels
        
        # Process through center block
        center = self.center(enc5)  # 2048 channels

        # Decode and concatenate
        dec5 = self.dec5(torch.cat([self.up(center, enc5.size()[-2:]), enc5], 1))  # 2048+2048 -> 512 channels
        dec4 = self.dec4(torch.cat([self.up(dec5, enc4.size()[-2:]), enc4], 1))  # 512+1024 -> 256 channels
        dec3 = self.dec3(torch.cat([self.up(dec4, enc3.size()[-2:]), enc3], 1))  # 256+512 -> 128 channels
        dec2 = self.dec2(torch.cat([self.up(dec3, enc2.size()[-2:]), enc2], 1))  # 128+256 -> 64 channels
        dec1 = self.dec1(torch.cat([self.up(dec2, enc1.size()[-2:]), enc1], 1))  # 64+64 -> 32 channels

        # Output layer
        out = self.out(self.up(dec1, x.size()[-2:]))  # 32 -> out_channels

        return out

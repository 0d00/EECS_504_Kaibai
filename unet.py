import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.step = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.step(x)
class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = ConvBlock(1, 64)
        self.layer2 = ConvBlock(64, 128)
        self.layer3 = ConvBlock(128, 256)
        self.layer4 = ConvBlock(256, 512)


        self.layer5 = ConvBlock(256+512, 256)
        self.layer6 = ConvBlock(128+256, 128)
        self.layer7 = ConvBlock(64+128, 64)


        self.layer8 = torch.nn.Conv2d(64, 1, kernel_size=1)


        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        x1 = self.layer1(x)
        x1_p = self.maxpool(x1)

        x2 = self.layer2(x1_p)
        x2_p = self.maxpool(x2)

        x3 = self.layer3(x2_p)
        x3_p = self.maxpool(x3)

        x4 = self.layer4(x3_p)

        x5 = self.upsample(x4)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.layer5(x5)

        x6 = self.upsample(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.layer6(x6)

        x7 = self.upsample(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.layer7(x7)


        x8 = self.layer8(x7)

        return x8
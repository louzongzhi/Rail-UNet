import torch
import torch.nn as nn


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class NestedUNet(nn.Module):
    def __init__(self, num_classes=9, input_channels=3, deep_supervision=False, n_channels=3, n_classes=9, bilinear=False, **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_blocks = nn.ModuleList([
            self._make_conv_block(input_channels, nb_filter[0]),
            self._make_conv_block(nb_filter[0], nb_filter[1]),
            self._make_conv_block(nb_filter[1], nb_filter[2]),
            self._make_conv_block(nb_filter[2], nb_filter[3]),
            self._make_conv_block(nb_filter[3], nb_filter[4])
        ])

        if self.deep_supervision:
            self.final_layers = nn.ModuleList([nn.Conv2d(nb_filter[0], num_classes, kernel_size=1) for _ in range(4)])
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        x = input
        outputs = []
        for i, conv_block in enumerate(self.conv_blocks):
            x = self.pool(x)
            x = conv_block(x)
            if i > 0:
                x = torch.cat([x, self.up(outputs[-1])], 1)
            outputs.append(x)

        if self.deep_supervision:
            return [self.final_layers[i](x) for i, x in enumerate(outputs)]
        else:
            return self.final(x)

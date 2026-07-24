import torch
import torch.nn as nn


class Block(nn.Module):
    """Residual block: two conv layers with a skip connection."""

    def __init__(self, ch_in, ch_out, downsample=False):
        super(Block, self).__init__()
        stride = 2 if downsample else 1

        self.main = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out)
        )

        self.skip = nn.Sequential()
        if downsample or ch_in != ch_out:
            self.skip = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(ch_out)
            )

        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.main(x) + self.skip(x))


class ResNet(nn.Module):
    """ResNet for multi-label solar cell defect detection (crack / inactive)."""

    def __init__(self):
        super(ResNet, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage1 = nn.Sequential(Block(64, 64),  Block(64, 64))
        self.stage2 = nn.Sequential(Block(64, 128, downsample=True),  Block(128, 128))
        self.stage3 = nn.Sequential(Block(128, 256, downsample=True), Block(256, 256))
        self.stage4 = nn.Sequential(Block(256, 512, downsample=True), Block(512, 512))

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)

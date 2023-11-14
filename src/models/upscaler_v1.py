import torch.nn as nn


class Upscaler(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = nn.BatchNorm2d(3)
        self.l2 = nn.Conv2d(3, 24, (3, 3), padding=1)
        self.l3 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.l4 = nn.Sigmoid()
        self.l5 = nn.BatchNorm2d(24)
        self.l6 = nn.Upsample(scale_factor=2)
        self.l7 = nn.Conv2d(24, 3, (3, 3), padding=1)
        self.l8 = nn.Sigmoid()
        self.l9 = nn.BatchNorm2d(3)
        self.l10 = nn.Upsample(scale_factor=2)
        self.l11 = nn.Sigmoid()

    def forward(self, x):
        y = self.l1(x)
        y = self.l2(y)
        y = self.l3(y)
        y = self.l4(y)
        y = self.l5(y)
        y = self.l6(y)
        y = self.l7(y)
        y = self.l8(y)
        y = self.l9(y)
        y = self.l10(y)

        return self.l11(y)

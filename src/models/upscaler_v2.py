import torch
import torch.nn as nn


class UpscalerV2(nn.Module):
    @staticmethod
    def BlockUp(input_dim, output_dim, conv_dim):
        return nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.Conv2d(input_dim, output_dim, (conv_dim, conv_dim), padding=(conv_dim - 1) // 2),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2)
        )

    @staticmethod
    def BlockConv(input_dim, output_dim):
        return nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.Conv2d(input_dim, output_dim, (3, 3), padding=1),
            nn.LeakyReLU()
        )

    def __init__(self):
        super().__init__()

        self.block_up_1_3x3 = self.BlockUp(3, 16, 3)
        self.block_up_1_5x5 = self.BlockUp(3, 16, 5)
        self.block_up_1_7x7 = self.BlockUp(3, 16, 7)
        self.block_up_1_9x9 = self.BlockUp(3, 16, 9)

        self.conv_1_1 = self.BlockConv(64, 32)
        self.conv_1_2 = self.BlockConv(32, 3)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_3x3 = self.block_up_1_3x3(x)
        x_5x5 = self.block_up_1_5x5(x)
        x_7x7 = self.block_up_1_7x7(x)
        x_9x9 = self.block_up_1_9x9(x)

        extended = torch.cat((x_3x3, x_5x5, x_7x7, x_9x9), dim=1)
        x_32x32 = self.conv_1_1(extended)
        output = self.conv_1_2(x_32x32)

        return self.sigmoid(output)

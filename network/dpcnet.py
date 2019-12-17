import torch
import torch.nn as nn
import math


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.1, inplace=True)
    )


def conv_basic(dropout, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    if dropout:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.PReLU(),
            nn.Dropout(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.PReLU()
        )


class DPCNet(nn.Module):
    def __init__(self, pose_type='quaternion'):
        super(DPCNet, self).__init__()
        # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)

        # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
        self.cnn1 = nn.Sequential(
            conv_basic(True, 3, 64, kernel_size=3, stride=2, padding=1),
            conv_basic(True, 64, 64, kernel_size=3, stride=2, padding=1),
            conv_basic(True, 64, 128, kernel_size=3, stride=1, padding=1),
        )
        self.cnn2 = nn.Sequential(
            conv_basic(True, 3, 64, kernel_size=3, stride=2, padding=1),
            conv_basic(True, 64, 64, kernel_size=3, stride=2, padding=1),
            conv_basic(True, 64, 128, kernel_size=3, stride=1, padding=1),
        )

        self.concat_net = nn.Sequential(
            conv_basic(True, 256, 256, kernel_size=3, stride=2, padding=1),
            conv_basic(True, 256, 512, kernel_size=3, stride=2, padding=1),
            conv_basic(True, 512, 1024, kernel_size=3, stride=2, padding=1),
            conv_basic(True, 1024, 4096, kernel_size=3, stride=2, padding=1),
            conv_basic(True, 4096, 4096, kernel_size=3, stride=2, padding=1),
            conv_basic(True, 4096, 7, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(7, 7, kernel_size=(1, 1), stride=1, padding=0)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, 0.5 / math.sqrt(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Initialize last conv weights to 0 to ensure the initial transform is Identity
        self.concat_net[-1].weight.data.zero_()

    def forward(self, img_1, img_2):

        x1 = self.cnn1(img_1)
        x2 = self.cnn2(img_2)

        x = torch.cat((x1, x2), 1)
        y = self.concat_net(x)
        y = y.view(-1, 7)
        return y[:,:4], y[:,4:]

import torch
import torch.nn as nn
import torch.nn.functional as F
from network.backbone.resnet_base import Bottleneck_elu
from network.backbone.resnet_base import *
from util import *

class ResNetHead(nn.Module):
    def __init__(self):
        super(ResNetHead, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.elu = nn.ELU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.maxpool(x)

        return x


class ResModule(nn.Module):

    def __init__(self, inplanes, planes, blocks_n, stride, layer_idx,  block=Bottleneck_elu):
        super(ResModule, self).__init__()
        self.module_name = 'layer'+str(layer_idx)
        self.inplanes = inplanes
        self.planes = planes

        self.resModule = nn.ModuleDict({
            self.module_name:  self._make_layer(
                block, self.planes, blocks_n, stride)
        })

        # self.__dict__.update(
        #     {self.module_name: self._make_layer(
        #         block, self.planes, blocks_n, stride)
        #      }
        # )
        # self.layer = self._make_layer(
        #     block, self.planes, blocks_n, stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.__dict__[self.module_name](x)
        # x = vars(self)[self.module_name](x)
        # x = self.layer(x)
        x = self.resModule[self.module_name](x)
        return x


class MultiNet(nn.Module):
    _inplanes = 64

    def __init__(self, pose_type='quaternion'):  # fc1_shape
        super(MultiNet, self).__init__()

        self.feature_resnet = resnet50(pretrained=True)
        self.feature_down = nn.Sequential(
                nn.Linear(2048, 512),
                nn.LeakyReLU(),
                nn.Dropout(0.5))
        # phase1
        self.regressor1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 7))
        # phase2
        self.lstm = nn.LSTM(input_size=1024, hidden_size=512, num_layers=3, batch_first=True, dropout=0.5, bidirectional=True)
        self.regressor2 = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 256),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 7))


    def forward(self, image1, image2, image3, image4, image5, phase=1):
        self.lstm.flatten_parameters()
        out1 = self.feature_down(self.feature_resnet(image1))
        out2 = self.feature_down(self.feature_resnet(image2))
        out3 = self.feature_down(self.feature_resnet(image3))
        out4 = self.feature_down(self.feature_resnet(image4))
        out5 = self.feature_down(self.feature_resnet(image5))

        lstm_in1 = torch.cat([out1, out2], dim=1)
        lstm_in2 = torch.cat([out2, out3], dim=1)
        lstm_in3 = torch.cat([out3, out4], dim=1)
        lstm_in4 = torch.cat([out4, out5], dim=1)

        outputs = []
        lstm_input_list = [lstm_in1, lstm_in2, lstm_in3, lstm_in4]
        # phase1
        if phase == 1:
            for i in range(len(lstm_input_list)):
                out = self.regressor1(lstm_input_list[i])
                outputs.append(out)
        else:
            # phase2
            lstm_in = torch.stack(lstm_input_list, 1)
            lstm_out = self.lstm(lstm_in)

            for i in range(lstm_out[0].size()[1]):
                out = self.regressor2(lstm_out[0][:, i, :])
                outputs.append(out)
        return outputs


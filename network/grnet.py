import torch
import torch.nn as nn
from network.backbone.resnet_base import Bottleneck_elu


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class GRNet(nn.Module):

    def __init__(self, pose_type='quaternion'):
        super(GRNet, self).__init__()
        block = Bottleneck_elu
        layers = [3, 4, 6, 3]
        num_classes = 1000
        zero_init_residual = False,
        groups = 1
        width_per_group = 64
        replace_stride_with_dilation = None
        norm_layer = None
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.pose_type = pose_type
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.elu = nn.ELU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.layer5 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm1 = nn.LSTM(input_size=2048, hidden_size=1000, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=1000, hidden_size=1000, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=2048, hidden_size=1000, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(49000, 1024)
        self.fc2 = nn.Linear(49000, 1024)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 4)
        self.fc5 = nn.Linear(1024, 3)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.CNN1 = nn.Sequential(self.conv1, self.bn1, self.elu, self.maxpool, self.layer1, self.layer2, self.layer3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck_elu):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        self.lstm3.flatten_parameters()
        #########CNN1###################
        out_CNN1 = self.CNN1(x1)
        out_CNN2 = self.CNN1(x2)
        #########RCNN1##################
        out_Res1_1 = self.layer5(torch.cat((out_CNN1, out_CNN2), 1))
        out_Res1_1_flat = torch.flatten(out_Res1_1, start_dim=2).permute(0, 2, 1)
        out_lstm1, _ = self.lstm1(out_Res1_1_flat)
        out_lstm2, _ = self.lstm2(out_lstm1)
        out_lstm2_flat = torch.flatten(out_lstm2, start_dim=1)
        out_fc1 = self.fc1(out_lstm2_flat)
        #########RCNN2#################
        out_Res2_1 = self.layer4(out_CNN2)
        out_Res2_1_flat = torch.flatten(out_Res2_1, start_dim=2).permute(0, 2, 1)
        out_lstm3, _ = self.lstm3(out_Res2_1_flat)
        out_lstm3_flat = torch.flatten(out_lstm3, start_dim=1)
        out_fc2 = self.fc2(out_lstm3_flat)
        #########FCFL##################
        out_fc3 = self.fc3(torch.cat((out_fc1, out_fc2), 1))
        rot_out = self.fc4(out_fc3)
        trans_out = self.fc5(out_fc3)

        return rot_out, trans_out

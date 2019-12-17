import torch
import torch.nn as nn
import torch.nn.functional as F
from network.backbone.resnet_base import Bottleneck_elu


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


class VLocNet(nn.Module):
    _inplanes = 64

    def __init__(self, pose_type='quaternion'):  # fc1_shape
        super(VLocNet, self).__init__()

        layers = [3, 4, 6, 3]
        strides = [1, 2, 2, 2]
        self.pose_type = pose_type
        self.block = Bottleneck_elu
        self.share_levels_n = 3
        self.dropout = 0.2
        self.recur_pose = True
        self.pooling_size = 1

        self.odom_en1_head = ResNetHead()
        self.odom_en2_head = ResNetHead()  # definitely share

        # odometry_encoder1
        _layers = []
        self.inplanes = self._inplanes
        for i in range(1, len(layers)):  # layer1..3 corresponding to res2..4 in paper
            planes = 64*2**(i-1)
            _layers.append(
                ResModule(inplanes=self.inplanes, planes=planes,
                          blocks_n=layers[i-1], stride=strides[i-1], layer_idx=i)
            )
            self.inplanes = planes * self.block.expansion
        self.odom_en1 = nn.Sequential(*_layers)

        # odometry_encoder2 and global_encoder: sharing parts
        _layers = []
        self.inplanes = self._inplanes
        for i in range(1, self.share_levels_n):  # corresponding to res2..share_levels_n in paper
            planes = 64*2**(i-1)
            _layers.append(
                ResModule(inplanes=self.inplanes, planes=planes,
                          blocks_n=layers[i-1], stride=strides[i-1], layer_idx=i)
            )
            self.inplanes = planes * self.block.expansion
        self._inplanes_r = self.inplanes  # save the results
        self.odom_en2_share = nn.Sequential(*_layers)


        # odometry_encoder2: rest parts
        _layers = []
        self.inplanes = self._inplanes_r
        for i in range(self.share_levels_n, len(layers)):
            planes = 64*2**(i-1)
            _layers.append(
                ResModule(inplanes=self.inplanes, planes=planes,
                          blocks_n=layers[i-1], stride=strides[i-1], layer_idx=i)
            )
            self.inplanes = planes * self.block.expansion

        self.odom_en2_sep = nn.Sequential(*_layers)

        # odom_final_res:
        self.odom_final_res = ResModule(inplanes=self.inplanes*2,
                                        planes=64*2**(len(layers)-1),
                                        blocks_n=layers[len(layers)-1], stride=strides[len(layers)-1], layer_idx=len(layers))


        self.odom_avgpool = nn.AdaptiveAvgPool2d(self.pooling_size)

        self.odom_fc1 = nn.Linear(
            2048*self.pooling_size*self.pooling_size, 1024)
        if self.pose_type[0] == 'q':
            self.odom_fcx = nn.Linear(1024, 3)
            self.odom_fcq = nn.Linear(1024, 4)
        elif self.pose_type[0] == 'd':
            self.odom_fcx = nn.Linear(1024, 3)
            self.odom_fcq = nn.Linear(1024, 3)
            self.odom_fca = nn.Linear(1024, 1)
        else:
            self.odom_fcx = nn.Linear(1024, 3)
            self.odom_fcq = nn.Linear(1024, 3)
            self.odom_fca = nn.Linear(1024, 1)


        self.odom_dropout = nn.Dropout(p=self.dropout)

    def forward(self, image1, image2):
        '''
            input: tuple(images,pose_p)
            images NxTx3xHxW, T=2+1 for now
            pose_p: NxTx7 ,previous poses except the current one; pose_p[0] is dummy
            return: Nx7, NxTx7
        '''

        out1 = self.odom_en1_head(image1)
        out1 = self.odom_en1(out1)  # previous frame

        out2 = self.odom_en2_head(image2)
        out2 = self.odom_en2_share(out2)  # current frame
        out2 = self.odom_en2_sep(out2)

        out2 = torch.cat([out1, out2], dim=1)
        out2 = self.odom_final_res(out2)
        out2 = self.odom_avgpool(out2)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.odom_fc1(out2)
        out2 = F.elu(out2)
        out2 = self.odom_dropout(out2)

        if self.pose_type[0] == 'q':
            trans_odom = self.odom_fcx(out2)
            rot_odom = F.tanh(self.odom_fcq(out2))
            outputs = [rot_odom, trans_odom]
        elif self.pose_type[0] == 'd':
            trans_odom = self.odom_fcx(out2)
            rot_odom = self.odom_fcq(out2)
            ang_odom = self.odom_fca(out2)
            outputs = [rot_odom, ang_odom, trans_odom]
        else:
            trans_odom = self.odom_fcx(out2)
            rot_odom = self.odom_fcq(out2)
            ang_odom = self.odom_fca(out2)
            outputs = [rot_odom, ang_odom, trans_odom]
        return outputs

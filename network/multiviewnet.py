import torch
import torch.nn as nn
from network.backbone.alexnet_base import *

class MultiViewNet(nn.Module):
    def __init__(self, pose_type='quaternion'):
        super(MultiViewNet, self).__init__()
        self.pose_type = pose_type
        self.init_fn = nn.init.xavier_normal_
        self.backbone = alexnet()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(2048, 7)
        ######initialization#######
        self.init_fn(self.fc.weight)

    def forward(self, im1, im2):
        feature1 = self.backbone(im1)
        feature2 = self.backbone(im2)
        concat_feature = torch.cat((feature1, feature2), 1)
        outputs = self.fc(self.relu(concat_feature))
        return outputs[:,:4], outputs[:,4:]
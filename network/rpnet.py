import torch
import torch.nn as nn
from network.backbone.googlenet_base import *

class RPNet(nn.Module):
    def __init__(self, pose_type='quaternion'):
        super(RPNet, self).__init__()
        self.pose_type = pose_type
        self.init_fn = nn.init.xavier_normal_
        self.backbone = googlenet(init_weights=True)
        self.fc_i = nn.Linear(256, 256)
        #self.bach_norm = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(0.5)
        self.fc_r = nn.Linear(256, 4)
        self.fc_t = nn.Linear(256, 3)
        #self.tanh = nn.Tanh()
        ######initialization#######
        self.init_fn(self.fc_i.weight)
        self.init_fn(self.fc_r.weight)
        self.init_fn(self.fc_t.weight)

    def forward(self, im1, im2):
        feature1 = self.backbone(im1)
        feature2 = self.backbone(im2)
        concat_feature = torch.cat((feature1, feature2), 1)
        fc1 = self.fc_i(concat_feature)
        #bn = self.bach_norm(fc1)
        active = self.relu(fc1)
        #drop = self.dropout(active)
        rot_out = self.fc_r(active)
        trans_out = self.fc_t(active)
        outputs = [rot_out, trans_out]
        return outputs
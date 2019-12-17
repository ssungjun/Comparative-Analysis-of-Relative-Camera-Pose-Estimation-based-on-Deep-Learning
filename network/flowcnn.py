import torch
import torch.nn as nn
from .backbone.flownet_base import FlowNetSD

class FlowCNN(nn.Module):
    def __init__(self, pose_type='quaternion'):
        super(FlowCNN, self).__init__()
        self.net = FlowNetSD(True)


    def forward(self, im1, im2):
        outputs = self.net(im1, im2)
        return outputs
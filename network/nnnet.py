import torch
import torch.nn as nn
from network.backbone.resnet_base import *

class nnNet(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, pose_type='quaternion'):
        super(nnNet, self).__init__()
        self.pose_type = pose_type
        self.init_fn = nn.init.xavier_normal_
        self.backbone = resnet34(pretrained=True, progress=True)#vgg11_bn(pretrained=True, progress=True)
        self.fc_i = nn.Linear(1024, 1024)
        self.bach_norm = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        if self.pose_type[0] == 'q':
            self.fc_r = nn.Linear(1024, 4)
            self.fc_t = nn.Linear(1024, 3)
        elif self.pose_type[0] == 'd':
            self.fc_r = nn.Linear(1024, 3)
            self.fc_a = nn.Linear(1024, 1)
            self.fc_t = nn.Linear(1024, 3)
        else:
            self.fc_r = nn.Linear(1024, 3)
            self.fc_a = nn.Linear(1024, 1)
            self.fc_t = nn.Linear(1024, 3)

        self.tanh = nn.Tanh()
        ######initialization#######
        self.init_fn(self.fc_i.weight)
        self.init_fn(self.fc_r.weight)
        self.init_fn(self.fc_t.weight)
        if self.pose_type[0] != 'q':
            self.init_fn(self.fc_a.weight)

    def forward(self, im1, im2):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        feature1 = self.backbone(im1)
        feature2 = self.backbone(im2)
        concat_feature = torch.cat((feature1, feature2), 1)
        fc1 = self.fc_i(concat_feature)
        bn = self.bach_norm(fc1)
        active = self.relu(bn)
        drop = self.dropout(active)
        rot_out = self.fc_r(drop)
        scaled_rot = rot_out#torch.div(rot_out.T, torch.sqrt(torch.sum(rot_out ** 2, 1))).T
        if self.pose_type[0] == 'q':
            trans_out = self.fc_t(drop)
            outputs = [scaled_rot, trans_out]
        elif self.pose_type[0] == 'd':
            trans_out = self.fc_t(drop)
            angle_out = self.fc_a(drop)
            outputs = [scaled_rot, angle_out, trans_out]
        else:
            trans_out = self.fc_t(drop)
            angle_out = self.fc_a(drop)
            outputs = [scaled_rot, angle_out, trans_out]
        return outputs
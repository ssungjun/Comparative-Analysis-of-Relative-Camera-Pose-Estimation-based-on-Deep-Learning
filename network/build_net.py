from operator import eq
from network.vlocnet import VLocNet
from network.nnnet import nnNet
from network.rpnet import RPNet
from network.dpcnet import DPCNet
from network.grnet import GRNet
from network.flowcnn import FlowCNN
from network.multiviewnet import MultiViewNet
from network.multi_net import MultiNet

def build(args):
    if eq(args.net, 'nnNet'):
        net = nnNet(args.pose_type)
    elif eq(args.net, 'VLocNet'):
        net = VLocNet(args.pose_type)
    elif eq(args.net, 'RPNet'):
        net = RPNet(args.pose_type)
    elif eq(args.net, 'DPCNet'):
        net = DPCNet(args.pose_type)
    elif eq(args.net, 'GRNet'):
        net = GRNet(args.pose_type)
    elif eq(args.net, 'FlowCNN'):
        net = FlowCNN(args.pose_type)
    elif eq(args.net, 'MultiViewNet'):
        net = MultiViewNet(args.pose_type)
    elif eq(args.net, 'multi'):
        net = MultiNet(args.pose_type)
    return net

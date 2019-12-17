import argparse
import os

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def give_parser():
    parser = argparse.ArgumentParser(
         description='Single Shot MultiBox Detector Training With Pytorch')
    train_set = parser.add_mutually_exclusive_group()

    parser.add_argument('--dataset_root', default='G:\\dataset\\',
                    help='Dataset root directory path')
    parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
    parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
    parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=1e-3, type=float,
                    help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.2, type=float,
                    help='Gamma update for SGD')
    parser.add_argument('--visdom', default=True, type=str2bool,
                    help='Use visdom for loss visualization')
    parser.add_argument('--save_folder', default='weight',
                    help='Directory for saving checkpoint models')
    parser.add_argument('--max_epoch', default=100, type=int,
                    help='Resume training at this iter')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
    parser.add_argument('--batch_size', default=18, type=int,
                    help='Batch size for training')
    parser.add_argument('--dataset', default='3dpw', choices=['3dpw', '7scenes', 'kitti'],
                    type=str, help='3dpw or 7scenes or kitti')
    parser.add_argument('--target_type', default='non_inv', choices=['non_inv', 'inv'],
                    type=str, help='non_inv or inv')
    parser.add_argument('--pose_type', default='quaternion', choices=['quaternion', 'degree', 'pose'],
                    type=str, help='quaternion or degree or pose')
    parser.add_argument('--seven_opt', default='total', choices=['total', 'chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs'],
                    type=str, help='7scenes sequence option')
    parser.add_argument('--net', default='multi', choices=['nnNet', 'VLocNet', 'RPNet', 'DPCNet',
                                                           'GRNet', 'FlowCNN', 'MultiViewNet', 'multi'],
                    type=str, help='build net')

    args = parser.parse_args()

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    return args

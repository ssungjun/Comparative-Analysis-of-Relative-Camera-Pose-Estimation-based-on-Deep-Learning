from data_loader_pose import *
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='jaad', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--input_type', default='rgb', type=str,
                    help='INput tyep default rgb options are [rgb,brox,fastOF]')
parser.add_argument('--dataset_root', default='G:\\dataset\\',
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=True, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weight\\',
                    help='Directory for saving checkpoint models')
parser.add_argument('--ssd_dim', default=300, type=int, help='Input Size for SSD')  # only support 300 now
parser.add_argument('--length', default=30, type=int,
                    help='sequence length')
parser.add_argument('--max_epoch', default=100, type=int,
                    help='Resume training at this iter')

args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    img_list = {}
    temp_list = []
    img_path = args.dataset_root + 'smartcar/expand_image'
    '''video_id = os.listdir(img_path)
    video_id.sort()
    for i in video_id:
        ped_id = os.listdir(os.path.join(img_path, i))
        ped_id.sort()
        for j in ped_id:
            frame_id = os.listdir(os.path.join(img_path, i, j))
            frame_id.sort()
            for k in frame_id:
                temp_list.append(os.path.join(img_path, i, j, k))

    for i in tqdm(temp_list, desc='image_read', mininterval=1):
        img_list[i] = cv2.imread(i)'''

    #if args.dataset_root != '/data/jaad/':
    #    parser.error('Must specify dataset if specifying dataset_root')
    args.train_sets = 'train'
    args.means = (104, 117, 123)
    #['3dpw', '7scenes', 'kitti']
    #['non_inv', 'inv']
    #['pose', 'degree', 'quaternion']
    #['total', 'chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
    dataset = PoseDetection(root=args.dataset_root, image_set='train', dataset_name='7scenes', target_type='non_inv', pose_type='quaternion', seven_scene_opt='total')
    testset = PoseDetection(root=args.dataset_root, image_set='test', dataset_name='7scenes', target_type='inv', pose_type='quaternion', seven_scene_opt='total')

    # train_dataset = JAADDetection(args.data_root, args.train_sets, SSDAugmentation(args.ssd_dim, args.means),
    #                              AnnotationTransform(), input_type=args.input_type)
    # val_dataset = JAADDetection(args.data_root, 'test', BaseTransform(args.ssd_dim, args.means),
    #                            AnnotationTransform(), input_type=args.input_type,
    #                            full_test=False)

    #if args.visdom:
    #    import visdom
    #    viz = visdom.Visdom()

    #ssd_net = build_net(size=300, length=args.length)
    # ssd_net = SSD300(depth=50, width=1, training='train')
    #net = ssd_net

    '''if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        # ssd_net.load_weights(args.resume)
        ssd_net.load_state_dict(torch.load(args.resume))
    # else:
    #    vgg_weights = torch.load(args.save_folder + args.basenet)
    #    print('Loading base network...')
    #    ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        net.module.lstm.apply(weights_init)
        net.module.lstm2.apply(weights_init)
        net.module.fc_action.apply(weights_init)
        net.module.fc_traffic.apply(weights_init)
        net.module.fc_crossing.apply(weights_init)
    # if not args.resume:
    #    print('Initializing weights...')
    #    # initialize newly added layers' weights with xavier method
    #    ssd_net.apply(weights_init)
    # ssd_net.multibox.apply(weights_init)
    # action_eta = nn.Parameter(torch.Tensor([0.0]).requires_grad_().cuda())
    # crossing_eta = nn.Parameter(torch.Tensor([0.0]).requires_grad_().cuda())
    # traffic_eta = nn.Parameter(torch.Tensor([0.0]).requires_grad_().cuda())
    # eta = [action_eta, crossing_eta, traffic_eta]

    # train_params = [{'params': net.parameters(), 'lr': args.lr},
    #                {'params': eta, 'lr': args.lr}]
    train_params = [{'params': net.parameters(), 'lr': args.lr}]
    optimizer = optim.Adam(train_params, weight_decay=args.weight_decay)
    #optimizer = optim.SGD(train_params, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = multi_task_loss

    net.train()'''
    # bonenet freeze
    # for param in net.module.bonenet.parameters():
    #    param.requires_grad = False
    # loss counters
    avg_size = 100
    conf_loss = 0
    cross_loss = 0
    traffic_loss = 0
    acc = 0
    best_acc = 0
    avg_conf_loss = np.zeros(avg_size)
    avg_cross_loss = np.zeros(avg_size)
    avg_traffic_loss = np.zeros(avg_size)
    avg_conf = 0
    avg_cross = 0
    avg_traffic = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    #print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    '''if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Conf Loss', 'Cross Loss', 'Traffic Loss', 'Acc']
        train_plot = create_vis_plot('train', 'Loss', vis_title, vis_legend, viz)
        test_plot = create_vis_plot('test', 'Loss', vis_title, vis_legend, viz)'''

    for iteration in range(args.max_epoch):
        print("%d epoch start" % (iteration))
        data_loader = data.DataLoader(dataset, args.batch_size,
                                      num_workers=0,
                                      shuffle=True, collate_fn=detection_collate,
                                      pin_memory=True)
        test_loader = data.DataLoader(testset, args.batch_size,
                                      num_workers=0,
                                      shuffle=False, collate_fn=detection_collate,
                                      pin_memory=True)
        # create batch iterator
        batch_iterator = iter(data_loader)
        test_iterator = iter(test_loader)
        # if args.visdom and iteration != 0 and (iteration % epoch_size == 0):

        # if iteration in cfg['lr_steps']:
        #    step_index += 1
        #    adjust_learning_rate(optimizer, args.gamma, step_index)
        #net.train()
        #for param in net.module.parameters():
        #    param.requires_grad = True
        for image1, image2, targets, idx in tqdm(batch_iterator, desc='train iteration', mininterval=1):
            if args.cuda:
                images = Variable(images.cuda())
                # targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
                targets = Variable(targets.cuda())
            else:
                images = Variable(images)
                # targets = [Variable(ann, volatile=True) for ann in targets]
                targets = Variable(targets)
            # forward
            net.zero_grad()
            optimizer.zero_grad()
            out = net(images)

            # backprop
            loss_action, loss_traffic, loss_cross = criterion(out, targets, args.length)
            # action_precision = torch.exp(-action_eta)
            # crossing_precision = torch.exp(-crossing_eta)
            # traffic_precision = torch.exp(-traffic_eta)
            # action_loss = loss_action * action_precision + action_eta
            # crossing_loss = loss_cross * crossing_precision + crossing_eta
            # traf_loss = loss_traffic * traffic_precision + traffic_eta
            loss = loss_cross
            loss.backward()
            optimizer.step()
            batch_size = len(batch_iterator)
            conf_loss += loss_action.data / batch_size  # [0]
            cross_loss += loss_cross.data / batch_size  # [0]
            traffic_loss += loss_traffic.data / batch_size  # [0]
            acc += accuracy(out, targets, 30) / batch_size
        update_vis_plot(epoch, conf_loss, cross_loss, traffic_loss, acc, train_plot,
                        'append', viz, epoch_size)
        # reset epoch loss counters
        conf_loss = 0
        cross_loss = 0
        traffic_loss = 0
        acc = 0
        outlist = []
        targetlist = []

        net.eval()
        for param in net.module.parameters():
            param.requires_grad = False
        for images, targets in tqdm(test_iterator, desc='test iteration', mininterval=1):
            if args.cuda:
                images = Variable(images.cuda())
                # targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
                targets = Variable(targets.cuda())
            else:
                images = Variable(images)
                # targets = [Variable(ann, volatile=True) for ann in targets]
                targets = Variable(targets)
            # forward
            out = net(images)

            # backprop
            loss_action, loss_traffic, loss_cross = criterion(out, targets, args.length)
            # action_precision = torch.exp(-action_eta)
            # crossing_precision = torch.exp(-crossing_eta)
            # traffic_precision = torch.exp(-traffic_eta)
            # action_loss = loss_action * action_precision + action_eta
            # crossing_loss = loss_cross * crossing_precision + crossing_eta
            # traf_loss = loss_traffic * traffic_precision + traffic_eta
            batch_size = len(test_iterator)
            conf_loss += loss_action.data / batch_size  # [0]
            cross_loss += loss_cross.data / batch_size  # [0]
            traffic_loss += loss_traffic.data / batch_size  # [0]
            acc += accuracy(out, targets, 30) / batch_size
            outlist += out[2].round()[:, :, 1].cpu().type(torch.IntTensor).view(-1).data.tolist()
            targetlist += targets[:, :, :, -1].view(-1).cpu().type(torch.IntTensor).tolist()
            '''if iteration < avg_size:
                avg_conf_loss[iteration] = loss_action.data
                avg_cross_loss[iteration] = loss_cross.data
                avg_traffic_loss[iteration] = loss_traffic.data
                avg_conf = avg_conf_loss.sum() / (iteration + 1)
                avg_cross = avg_cross_loss.sum() / (iteration + 1)
                avg_traffic = avg_traffic_loss.sum() / (iteration + 1)
            else:
                avg_conf_loss = np.roll(avg_conf_loss, -1)
                avg_cross_loss = np.roll(avg_cross_loss, -1)
                avg_traffic_loss = np.roll(avg_traffic_loss, -1)
                avg_conf_loss[-1] = loss_action.data
                avg_cross_loss[-1] = loss_cross.data
                avg_traffic_loss[-1] = loss_traffic.data
                avg_conf = avg_conf_loss.sum() / avg_size
                avg_cross = avg_cross_loss.sum() / avg_size
                avg_traffic = avg_traffic_loss.sum() / avg_size
            if iteration % 10 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data), end=' ')
                print("acc = %.4f" % (accuracy(out, targets, args.length, True)))

            if args.visdom:
                update_vis_plot(iteration, loss_action, loss_cross, loss_traffic, avg_conf, avg_cross,
                                avg_traffic, iter_plot, 'append', viz)'''
        update_vis_plot(epoch, conf_loss, cross_loss, traffic_loss, acc, test_plot,
                        'append', viz, epoch_size)
        if best_acc < acc:
            best_acc = acc
            best_epoch = iteration
            torch.save(net.state_dict(), 'weights/ssd300_COCO_best.pth')
            print("%d epoch is best, acc = %f" % (best_epoch, best_acc))
            plot_confusion_matrix(outlist, targetlist,
                                  ['non_crossing', 'crossing'],
                                  'best_epoch.png', normalize=True)
        # reset epoch loss counters
        conf_loss = 0
        cross_loss = 0
        traffic_loss = 0
        acc = 0
        epoch += 1

        '''   if iteration != 0 and iteration % 5000 == 0:
                print('Saving state, iter:', iteration)
                torch.save(net.state_dict(), 'weights/ssd300_COCO_' +
                           repr(iteration) + '.pth')
        torch.save(net.state_dict(),
                   args.save_folder + '' + args.dataset + '.pth')'''


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        if type(m.bias) != type(None):
            m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend, viz):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 4)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, conf, cross, traffic, acc, window, update_type,
                    viz, epoch_size=1):
    if iteration != 0:
        viz.line(
            X=torch.ones((1, 4)).cpu() * iteration,
            Y=torch.Tensor(
                [conf, cross, traffic, acc]).unsqueeze(
                0).cpu(),
            win=window,
            update=update_type
        )
    else:
        viz.line(
            X=torch.zeros((1, 4)).cpu(),
            Y=torch.Tensor(
                [conf, cross, traffic, acc]).unsqueeze(
                0).cpu(),
            win=window,
            update=True
        )


def multi_task_loss(output, target, length):
    temp = target[:, 0, 0, -1].cpu().type(torch.IntTensor)
    onehot = np.eye(2)[temp]
    loss_action = F.binary_cross_entropy_with_logits(output[0][:, 0].contiguous(),
                                                     Variable(target[:, -1, 0, :8], requires_grad=False).contiguous(),
                                                     reduction='mean', pos_weight=torch.ones([8]) * 8)
    loss_traffic = F.binary_cross_entropy_with_logits(output[1][:, 0].contiguous(), Variable(target[:, -1, 0, 8:-1],
                                                                                             requires_grad=False).contiguous(),
                                                      reduction='mean', pos_weight=torch.ones([6]) * 6)
    loss_crossing = F.binary_cross_entropy(output[2][:, 0].contiguous(),
                                           Variable(torch.from_numpy(onehot).type(torch.cuda.FloatTensor),
                                                    requires_grad=False).cuda().contiguous(),
                                           reduction='mean')
    for i in range(1, length):
        temp = target[:, i, 0, -1].cpu().type(torch.IntTensor)
        onehot = np.eye(2)[temp]
        loss_action += F.binary_cross_entropy_with_logits(output[0][:, i].clone().contiguous(),
                                                          Variable(target[:, i, 0, :8],
                                                                   requires_grad=False).contiguous(),
                                                          reduction='mean', pos_weight=torch.ones([8]) * 8)
        loss_traffic += F.binary_cross_entropy_with_logits(output[1][:, i].clone().contiguous(),
                                                           Variable(target[:, i, 0, 8:-1],
                                                                    requires_grad=False).contiguous(),
                                                           reduction='mean', pos_weight=torch.ones([6]) * 6)
        loss_crossing += F.binary_cross_entropy(output[2][:, i].clone().contiguous(),
                                                Variable(torch.from_numpy(onehot).type(torch.cuda.FloatTensor),
                                                         requires_grad=False).cuda().contiguous(),
                                                reduction='mean')

    return loss_action / length, loss_traffic / length, loss_crossing / length


def accuracy(output, target, length, many_to_many=True):
    total_len = 0
    total_correct = 0
    if many_to_many:
        for i in range(length):
            total = output[2][:, i, -1].round().cpu().type(torch.IntTensor) == target[:, i, 0, -1].cpu().type(
                torch.IntTensor)
            total_len += len(total)
            total_correct += int(total.sum())
    else:
        for i in range(0, length, length - 1):
            total = output[2][:, i, -1].round().cpu().type(torch.IntTensor) == target[:, -1, 0, -1].cpu().type(
                torch.IntTensor)
            total_len += len(total)
            total_correct += int(total.sum())
    return float(total_correct) / total_len


def plot_confusion_matrix(output,target,
                          target_names, path,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    cm = np.zeros((2, 2))
    for i in range(len(target)):
        cm[int(target[i]), int(output[i])] += 1

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(path)


if __name__ == '__main__':
    train()

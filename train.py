import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from network.build_net import build
from util import *
from data_loader_pose import *
from argparser import give_parser
from eval import eval
from visualize import visualize


args = give_parser()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def train():
    args.train_sets = 'train'
    args.means = (104, 117, 123)
    datalist = pre_read_data(args)
    dataset = PoseDetection(root=args.dataset_root, image_set='train', dataset_name=args.dataset, target_type=args.target_type, pose_type=args.pose_type, seven_scene_opt=args.seven_opt, data=datalist)
    testset = PoseDetection(root=args.dataset_root, image_set='test', dataset_name=args.dataset, target_type=args.target_type, pose_type=args.pose_type, seven_scene_opt=args.seven_opt, data=datalist)
    '''dataset = PoseDetection(root=args.dataset_root, image_set='train', dataset_name=args.dataset,
                            target_type='non_inv', pose_type=args.pose_type, seven_scene_opt=args.seven_opt,
                            data=datalist)
    testset = PoseDetection(root=args.dataset_root, image_set='test', dataset_name=args.dataset,
                            target_type='inv', pose_type=args.pose_type, seven_scene_opt=args.seven_opt,
                            data=datalist)
    dataset = PoseDetection(root=args.dataset_root, image_set='train', dataset_name=args.dataset,
                            target_type='non_inv', pose_type='degree', seven_scene_opt=args.seven_opt,
                            data=datalist)
    testset = PoseDetection(root=args.dataset_root, image_set='test', dataset_name=args.dataset,
                            target_type='inv', pose_type='degree', seven_scene_opt=args.seven_opt,
                            data=datalist)'''

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    net = build(args)


    if args.cuda:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        net.load_state_dict(torch.load(args.resume))


    if args.cuda:
        net = net.cuda()


    train_params = [{'params': net.parameters(), 'lr': args.lr}]
    optimizer = optim.Adam(train_params, weight_decay=args.weight_decay)
    #optimizer = optim.SGD(train_params, momentum=args.momentum, weight_decay=args.weight_decay)
    von_loss = von_mises_loss
    mse_loss = nn.MSELoss()

    rot_loss = 0
    ang_loss = 0
    trans_loss = 0
    rot_err = 0
    ang_err = 0
    quat_err = 0
    quat_total_err = 0
    trans_err = 0
    best_rot = 999.99
    best_ang = 9999.99
    best_quat = 9999.99
    best_trans = 9999.99
    best_total = 999999.9
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = args.net + '_' + args.dataset + '_' + args.pose_type
        if args.pose_type[0] == 'q':
            vis_legend = ['Rotation Loss', 'Transport Loss', 'Rotation error', 'Transport error', 'Quaternion error']
        elif args.pose_type[0] == 'd':
            vis_legend = ['Rotation Loss', 'Angle Loss', 'Transport Loss', 'Rotation error', 'Angle error', 'Transport error']
        train_plot = create_vis_plot('train', 'Loss', vis_title, vis_legend, viz)
        test_plot = create_vis_plot('test', 'Loss', vis_title, vis_legend, viz)

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
        lr_decay = [int(args.max_epoch/5),int(2*args.max_epoch/5),int(3*args.max_epoch/5),int(4*args.max_epoch/5)]
        if iteration in lr_decay:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
        net.train()
        for param in net.module.parameters():
            param.requires_grad = True
        batch_size = len(batch_iterator)
        for batch in tqdm(batch_iterator, desc='train iteration', mininterval=1):
            if args.pose_type[0] == 'd':
                image1, image2, target1, target2, target3, idx = batch
            elif args.pose_type[0] == 'q':
                image1, image2, target1, target2, idx = batch
            if args.cuda:
                image1 = Variable(image1.cuda())
                image2 = Variable(image2.cuda())
                target1 = Variable(target1.cuda())
                target2 = Variable(target2.cuda())
                if args.pose_type[0] == 'd':
                    target3 = Variable(target3.cuda())
            else:
                image1 = Variable(image1)
                image2 = Variable(image2)
                target1 = Variable(target1)
                target2 = Variable(target2)
                if args.pose_type[0] == 'd':
                    target3 = Variable(target3)
            # forward
            net.zero_grad()
            optimizer.zero_grad()
            out = net(image1, image2)
            if args.pose_type[0] == 'd':
                angle_err = float(torch.mean(torch.abs(target2 - out[1])).data.cpu())
            elif args.pose_type[0] == 'q':
                rot_pred = quaternion_to_rotation(out[0])
                rot_gt = quaternion_to_rotation(target1)
                angle_err = rotation_error(rot_pred, rot_gt)
                quat_err = quaternion_error(out[0], target1)
            # backprop
            if args.pose_type[0] == 'd':
                loss_angle = von_loss(out[1], target2)
                loss_rotation = mse_loss(5*out[0], 5*target1)
                loss_transport = mse_loss(5*out[2], 5*target3)
                #loss = loss_rotation + loss_transport + loss_angle
                loss = torch.pow(loss_rotation*loss_transport*loss_angle, 1/3)
            elif args.pose_type[0] == 'q':
                '''if epoch < 10:
                    loss_rotation = orientation_loss(target1, out[0], 'non')
                else:
                    loss_rotation = orientation_loss(target1, out[0], 'pow')'''
                loss_rotation = mse_loss(out[0], target1)
                loss_transport = mse_loss(out[1], target2)
                loss = loss_rotation + loss_transport

            loss.backward()
            optimizer.step()
            rot_loss += loss_rotation.data / batch_size  # [0]
            trans_loss += loss_transport.data / batch_size  # [0]
            if args.pose_type[0] == 'q':
                rot_err += angle_err / batch_size
                quat_total_err += quat_err / batch_size
                trans_err += float(torch.mean(F.pairwise_distance(target2, out[1])).data.cpu()) / batch_size
            elif args.pose_type[0] == 'd':
                ang_loss += loss_angle.data / batch_size  # [0]
                ang_err += angle_err / batch_size
                rot_err += float(torch.mean(F.pairwise_distance(target1, out[0])).data.cpu()) / batch_size
                trans_err += float(torch.mean(F.pairwise_distance(target3, out[2])).data.cpu()) / batch_size

        if args.visdom:
            if args.pose_type[0] == 'q':
                datas = [rot_loss, trans_loss, rot_err, trans_err, quat_total_err]
            elif args.pose_type[0] == 'd':
                datas = [rot_loss, ang_loss, trans_loss, rot_err, ang_err, trans_err]
            update_vis_plot(epoch, datas, train_plot, 'append', viz, epoch_size)
        # reset epoch loss counters
        rot_loss = 0
        ang_loss = 0
        trans_loss = 0
        rot_err = 0
        ang_err = 0
        trans_err = 0
        quat_err = 0
        quat_total_err = 0

        net.eval()
        batch_size = len(test_iterator)
        for param in net.module.parameters():
            param.requires_grad = False
        for batch in tqdm(test_iterator, desc='test iteration', mininterval=1):
            if args.pose_type[0] == 'd':
                image1, image2, target1, target2, target3, idx = batch
            elif args.pose_type[0] == 'q':
                image1, image2, target1, target2, idx = batch
            if args.cuda:
                image1 = Variable(image1.cuda())
                image2 = Variable(image2.cuda())
                target1 = Variable(target1.cuda())
                target2 = Variable(target2.cuda())
                if args.pose_type[0] == 'd':
                    target3 = Variable(target3.cuda())
            else:
                image1 = Variable(image1)
                image2 = Variable(image2)
                target1 = Variable(target1)
                target2 = Variable(target2)
                if args.pose_type[0] == 'd':
                    target3 = Variable(target3)
            # forward
            net.zero_grad()
            out = net(image1, image2)
            if args.pose_type[0] == 'd':
                angle_err = float(torch.mean(torch.abs(target2 - out[1])).data.cpu())
            elif args.pose_type[0] == 'q':
                rot_pred = quaternion_to_rotation(out[0])
                rot_gt = quaternion_to_rotation(target1)
                angle_err = rotation_error(rot_pred, rot_gt)
                quat_err = quaternion_error(out[0],target1)
            # backprop
            if args.pose_type[0] == 'd':
                loss_angle = von_loss(out[1], target2)
                loss_rotation = mse_loss(5*out[0], 5*target1)
                loss_transport = mse_loss(5*out[2], 5*target3)

            elif args.pose_type[0] == 'q':
                loss_rotation = mse_loss(out[0], target1)
                #loss_rotation = orientation_loss(target1, out[0])
                loss_transport = mse_loss(out[1], target2)


            rot_loss += loss_rotation.data / batch_size  # [0]
            trans_loss += loss_transport.data / batch_size  # [0]
            if args.pose_type[0] == 'q':
                rot_err += angle_err / batch_size
                quat_total_err += quat_err / batch_size
                trans_err += float(torch.mean(F.pairwise_distance(target2, out[1])).data.cpu()) / batch_size
            elif args.pose_type[0] == 'd':
                ang_loss += loss_angle.data / batch_size  # [0]
                ang_err += angle_err / batch_size
                rot_err += float(torch.mean(F.pairwise_distance(target1, out[0])).data.cpu()) / batch_size
                trans_err += float(torch.mean(F.pairwise_distance(target3, out[2])).data.cpu()) / batch_size
        if best_rot > rot_err:
            save_path = os.path.join(args.save_folder,args.net+'_'+args.dataset+'_'+args.pose_type+'_'+args.seven_opt+'_best_rot.pth')
            torch.save(net.state_dict(), save_path)
            best_rot = rot_err
        if args.pose_type[0] == 'd':
            if best_ang > ang_err:
                save_path = os.path.join(args.save_folder,args.net+'_'+args.dataset+'_'+args.pose_type+'_'+args.seven_opt+'_best_ang.pth')
                torch.save(net.state_dict(), save_path)
                best_ang = ang_err
        if best_trans > trans_err:
            save_path = os.path.join(args.save_folder,
                                     args.net + '_' + args.dataset + '_' + args.pose_type+'_' + args.seven_opt + '_best_trans.pth')
            torch.save(net.state_dict(), save_path)
            best_trans = trans_err
        if args.pose_type[0] == 'q':
            total_err = trans_err + rot_err
        elif args.pose_type[0] == 'd':
            total_err = trans_err + rot_err + ang_err
        if best_total > total_err:
            save_path = os.path.join(args.save_folder,
                                     args.net + '_' + args.dataset + '_' + args.seven_opt + '_best_total.pth')
            torch.save(net.state_dict(), save_path)
            best_total = total_err
        if best_quat > quat_total_err:
            save_path = os.path.join(args.save_folder,
                                     args.net + '_' + args.dataset + '_' + args.seven_opt + '_best_quat.pth')
            torch.save(net.state_dict(), save_path)
            best_quat = quat_total_err
        if args.visdom:
            if args.pose_type[0] == 'q':
                datas = [rot_loss, trans_loss, rot_err, trans_err, quat_total_err]
            elif args.pose_type[0] == 'd':
                datas = [rot_loss, ang_loss, trans_loss, rot_err, ang_err, trans_err]
            update_vis_plot(epoch, datas, test_plot, 'append', viz, epoch_size)

        # reset epoch loss counters
        rot_loss = 0
        ang_loss = 0
        trans_loss = 0
        rot_err = 0
        ang_err = 0
        trans_err = 0
        quat_err = 0
        quat_total_err = 0
        epoch += 1
        print('best rot err: %f'%(best_rot))
        print('best trans err: %f'%(best_trans))
        print('best quat err: %f'%(best_quat))
        if args.pose_type[0] == 'd':
            print('best ang err: %f' % (best_ang))
        print('best total err: %f' % (best_total))


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
    legend_len = len(_legend)
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, legend_len)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, data, window, update_type, viz, epoch_size=1):
    data_len = len(data)
    if iteration != 0:
        viz.line(
            X=torch.ones((1, data_len)).cpu() * iteration,
            Y=torch.Tensor(
                data).unsqueeze(
                0).cpu(),
            win=window,
            update=update_type
        )
    else:
        viz.line(
            X=torch.zeros((1, data_len)).cpu(),
            Y=torch.Tensor(
                data).unsqueeze(
                0).cpu(),
            win=window,
            update=True
        )


def von_mises_loss(angle1, angle2):
    k = 10
    rad1 = angle1 * float(np.pi) / 180.0
    rad2 = angle2 * float(np.pi) / 180.0
    rad1_cos = torch.cos(rad1)
    rad1_sin = torch.sin(rad1)
    rad2_cos = torch.cos(rad2)
    rad2_sin = torch.sin(rad2)
    loss = torch.mean(1 - torch.exp(k*(rad1_cos*rad2_cos + rad1_sin*rad2_sin - 1)))
    return loss


def orientation_loss(quat_t, quat_p, mode='non'):
    betta = 0.1
    norm_quat_p = torch.sqrt(torch.sum(quat_p ** 2, 1))
    scaled_quat_p = torch.div(quat_p.T, norm_quat_p).T
    if mode =='pow':
        LG = 1 - torch.pow(torch.sum(quat_t*scaled_quat_p, 1), 2)
    else:
        LG = 1 - torch.sum(quat_t * scaled_quat_p, 1)
    LN = betta * torch.pow((1 - norm_quat_p), 2)
    loss = torch.mean(LG + LN)
    return loss


if __name__ == '__main__':
    train()
    best_type = 'total'  # ['rot', 'ang', 'trans', 'total']
    eval(best_type)
    visualize(args, best_type)

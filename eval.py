import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from network.build_net import build
from util import *
from data_loader_pose import *
from argparser import give_parser
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


def eval(best_type='total'):
    args.train_sets = 'test'
    args.means = (104, 117, 123)
    datalist = pre_read_data(args)
    testset = PoseDetection(root=args.dataset_root, image_set='test', dataset_name=args.dataset, target_type=args.target_type, pose_type=args.pose_type, seven_scene_opt=args.seven_opt, data=datalist)


    net = build(args)


    if args.cuda:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    if best_type[:2] != 'to':
        model_name = args.net + '_' + args.dataset + '_' + args.pose_type + '_' + args.seven_opt + '_best_'+best_type+'.pth'
    else:
        model_name = args.net + '_' + args.dataset + '_' + args.seven_opt + '_best_' + best_type + '.pth'
    save_path = os.path.join(args.save_folder, model_name)
    net.load_state_dict(torch.load(save_path))

    eval_path = os.path.join('output', args.net + '_' + args.dataset + '_' + args.seven_opt + '_' + best_type +'_eval')
    if not os.path.exists('output'):
        os.mkdir('output')
    if not os.path.exists(eval_path):
        os.mkdir(eval_path)

    if args.cuda:
        net = net.cuda()


    rot_err = 0
    trans_err = 0
    print('Loading the dataset...')

    print('Using the specified args:')
    print(args)

    test_loader = data.DataLoader(testset, args.batch_size,
                                  num_workers=0,
                                  shuffle=False, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    test_iterator = iter(test_loader)

    net.eval()
    batch_size = len(test_iterator)
    for param in net.module.parameters():
        param.requires_grad = False
    for image1, image2, target1, target2, idx in tqdm(test_iterator, desc='test iteration', mininterval=1):
        if args.cuda:
            image1 = Variable(image1.cuda())
            image2 = Variable(image2.cuda())
            target1 = Variable(target1.cuda())
            target2 = Variable(target2.cuda())
        else:
            image1 = Variable(image1)
            image2 = Variable(image2)
            target1 = Variable(target1)
            target2 = Variable(target2)
        # forward
        out = net(image1, image2)
        quat_gt = target1.data.cpu().numpy()
        quat_pred = out[0].data.cpu().numpy()
        norm_quat_p = np.sqrt(np.sum(quat_pred ** 2, 1))
        for i in range(quat_pred.shape[0]):
            quat_pred[i] = np.divide(quat_pred[i], norm_quat_p[i])
        trans_gt = target2.data.cpu().numpy()
        trans_pred = out[1].data.cpu().numpy()

        for i in range(len(idx)):
            file_name = '%06d_eval.txt'%(idx[i])
            with open(os.path.join(eval_path, file_name), 'wb') as f:
                f.write(('gt quat: %f, %f, %f, %f\n' % (quat_gt[i, 0], quat_gt[i, 1], quat_gt[i, 2], quat_gt[i, 3])).encode())
                f.write(('pred quat: %f, %f, %f, %f\n' % (quat_pred[i, 0], quat_pred[i, 1], quat_pred[i, 2], quat_pred[i, 3])).encode())
                f.write(('gt trans: %f, %f, %f\n' % (trans_gt[i, 0], trans_gt[i, 1], trans_gt[i, 2])).encode())
                f.write(('pred trans: %f, %f, %f' % (trans_pred[i, 0], trans_pred[i, 1], trans_pred[i, 2])).encode())

        rot_pred = quaternion_to_rotation(out[0])
        rot_gt = quaternion_to_rotation(target1)
        angle_err = rotation_error(rot_pred, rot_gt)

        rot_err += angle_err / batch_size
        trans_err += float(torch.mean(F.pairwise_distance(target2, out[1])).data.cpu()) / batch_size


    print('rot err: %f'%(rot_err))
    print('trans err: %f'%(trans_err))


if __name__ == '__main__':
    best_type = 'rot'  # ['rot', 'ang', 'trans', 'total']
    eval(best_type)
    visualize(args, best_type)

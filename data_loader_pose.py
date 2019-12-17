import os
import os.path
import torch
import torch.utils.data as data
import cv2, pickle
import numpy as np


def readsplitfile(splitfile):
    with open(splitfile, 'r') as f:
        temptrainvideos = f.readlines()
    trainvideos = []
    for vid in temptrainvideos:
        vid = vid.rstrip('\n')
        trainvideos.append(vid)
    return trainvideos


def pose_make_lists(dataset_path=None, dataset_name='3dpw', target_type='non_inv', pose_type='quaternion', seven_scene_opt='total'):
    data_path = os.path.join(dataset_path, dataset_name)
    end_path = data_path

    # make video_list
    if not dataset_name[0] == '7':
        trainlistpth = os.path.join(data_path, 'Train_file_list.txt')
        testlistpth = os.path.join(data_path, 'Test_file_list.txt')
        totallistpth = os.path.join(data_path, 'Total_file_list.txt')
    else:
        if seven_scene_opt[0] == 't':
            trainlistpth = os.path.join(data_path, 'Train_file_list.txt')
            testlistpth = os.path.join(data_path, 'Test_file_list.txt')
            totallistpth = os.path.join(data_path, 'Total_file_list.txt')
        elif seven_scene_opt[0] == 'c':
            trainlistpth = os.path.join(data_path, 'chess', 'Train_file_list.txt')
            testlistpth = os.path.join(data_path, 'chess', 'Test_file_list.txt')
            totallistpth = os.path.join(data_path, 'chess', 'Total_file_list.txt')
            end_path = os.path.join(end_path, 'chess')
        elif seven_scene_opt[0] == 'f':
            trainlistpth = os.path.join(data_path, 'fire', 'Train_file_list.txt')
            testlistpth = os.path.join(data_path, 'fire', 'Test_file_list.txt')
            totallistpth = os.path.join(data_path, 'fire', 'Total_file_list.txt')
            end_path = os.path.join(end_path, 'fire')
        elif seven_scene_opt[0] == 'h':
            trainlistpth = os.path.join(data_path, 'heads', 'Train_file_list.txt')
            testlistpth = os.path.join(data_path, 'heads', 'Test_file_list.txt')
            totallistpth = os.path.join(data_path, 'heads', 'Total_file_list.txt')
            end_path = os.path.join(end_path, 'heads')
        elif seven_scene_opt[0] == 'o':
            trainlistpth = os.path.join(data_path, 'office', 'Train_file_list.txt')
            testlistpth = os.path.join(data_path, 'office', 'Test_file_list.txt')
            totallistpth = os.path.join(data_path, 'office', 'Total_file_list.txt')
            end_path = os.path.join(end_path, 'office')
        elif seven_scene_opt[0] == 'p':
            trainlistpth = os.path.join(data_path, 'pumpkin', 'Train_file_list.txt')
            testlistpth = os.path.join(data_path, 'pumpkin', 'Test_file_list.txt')
            totallistpth = os.path.join(data_path, 'pumpkin', 'Total_file_list.txt')
            end_path = os.path.join(end_path, 'pumpkin')
        elif seven_scene_opt[0] == 'r':
            trainlistpth = os.path.join(data_path, 'redkitchen', 'Train_file_list.txt')
            testlistpth = os.path.join(data_path, 'redkitchen', 'Test_file_list.txt')
            totallistpth = os.path.join(data_path, 'redkitchen', 'Total_file_list.txt')
            end_path = os.path.join(end_path, 'redkitchen')
        elif seven_scene_opt[0] == 's':
            trainlistpth = os.path.join(data_path, 'stairs', 'Train_file_list.txt')
            testlistpth = os.path.join(data_path, 'stairs', 'Test_file_list.txt')
            totallistpth = os.path.join(data_path, 'stairs', 'Total_file_list.txt')
            end_path = os.path.join(end_path, 'stairs')

    if (target_type[:3] == 'non') and (pose_type[:3] == 'qua'):
        label_folder = 'relative_quaternion_pose'
    elif (target_type[:3] == 'non') and (pose_type[:3] == 'deg'):
        label_folder = 'relative_degree_pose'
    elif (target_type[:3] == 'non') and (pose_type[:3] == 'pos'):
        label_folder = 'relative_pose'
    elif (target_type[:3] == 'inv') and (pose_type[:3] == 'qua'):
        label_folder = 'inverse_relative_quaternion_pose'
    elif (target_type[:3] == 'inv') and (pose_type[:3] == 'deg'):
        label_folder = 'inverse_relative_degree_pose'
    elif (target_type[:3] == 'inv') and (pose_type[:3] == 'pos'):
        label_folder = 'inverse_relative_pose'

    with open(trainlistpth, 'r') as f:
        train_seq = f.readlines()
    with open(testlistpth, 'r') as f:
        test_seq = f.readlines()
    with open(totallistpth, 'r') as f:
        video_list = f.readlines()


    if not os.path.isfile(os.path.join(end_path,label_folder+'_list.pkl')):
        # make trainlist
        trainlist = []
        for i in train_seq:
            if dataset_name[0] == '3':
                frame_num = 'frame_%05d.txt'%(int(i.split('.')[0][-5:]))
                label_path = os.path.join(data_path, label_folder, i.split('\\')[1], frame_num)
                first_img = i[:-1]
                second_img = first_img.split('.')[0][:-5]+'%05d.jpg'%(int(first_img.split('.')[0][-5:]) + 1)
            elif dataset_name[0] == '7':
                frame_num = i.split('\\')[3].split('.')[0] + '.pose.txt'
                label_path = os.path.join(data_path, i.split('\\')[0], i.split('\\')[1], label_folder, frame_num)
                first_img = i[:-1]
                second_img = first_img.split('.')[0][:-6] + '%06d.color.png' % (int(first_img.split('.')[0][-6:]) + 1)
            else:
                frame_num = i.split('\\')[3].split('.')[0] + '.txt'
                label_path = os.path.join(data_path, label_folder, i.split('\\')[1], frame_num)
                first_img = i[:-1]
                second_img = first_img.split('.')[0][:-6] + '%06d.png' % (int(first_img.split('.')[0][-6:]) + 1)

            for idx, k in enumerate(video_list):
                if k == i:
                    vid = idx
                    break
            with open(label_path, 'r') as f:
                objects = f.readlines()
            label = []
            if pose_type[:3] == 'pos':
                label = np.zeros((3,4))
                for x in range(3):
                    for y in range(4):
                        label[x,y] = float(objects[x][y])
            elif pose_type[:3] == 'qua':
                label = [[float(objects[0].split()[0]),float(objects[0].split()[1]),float(objects[0].split()[2]), float(objects[0].split()[3])],
                         [float(objects[0].split()[4]), float(objects[0].split()[5]), float(objects[0].split()[6])]]
            elif pose_type[:3] == 'deg':
                label = [[float(objects[0].split()[0]),float(objects[0].split()[1]),float(objects[0].split()[2])], [float(objects[0].split()[3])],
                         [float(objects[0].split()[4]), float(objects[0].split()[5]), float(objects[0].split()[6])]]
            trainlist.append([vid, first_img, second_img, label])

        # make testlist
        testlist = []
        for i in test_seq:
            if dataset_name[0] == '3':
                frame_num = 'frame_%05d.txt' % (int(i.split('.')[0][-5:]))
                label_path = os.path.join(data_path, label_folder, i.split('\\')[1], frame_num)
                first_img = i[:-1]
                second_img = first_img.split('.')[0][:-5] + '%05d.jpg' % (int(first_img.split('.')[0][-5:]) + 1)
            elif dataset_name[0] == '7':
                frame_num = i.split('\\')[3].split('.')[0] + '.pose.txt'
                label_path = os.path.join(data_path, i.split('\\')[0], i.split('\\')[1], label_folder, frame_num)
                first_img = i[:-1]
                second_img = first_img.split('.')[0][:-6] + '%06d.color.png' % (int(first_img.split('.')[0][-6:]) + 1)
            else:
                frame_num = i.split('\\')[3].split('.')[0] + '.txt'
                label_path = os.path.join(data_path, label_folder, i.split('\\')[1], frame_num)
                first_img = i[:-1]
                second_img = first_img.split('.')[0][:-6] + '%06d.png' % (int(first_img.split('.')[0][-6:]) + 1)

            for idx, k in enumerate(video_list):
                if k == i:
                    vid = idx
                    break
            with open(label_path, 'r') as f:
                objects = f.readlines()
            label = []
            if pose_type[:3] == 'pos':
                label = np.zeros((3, 4))
                for x in range(3):
                    for y in range(4):
                        label[x, y] = float(objects[x][y])
            elif pose_type[:3] == 'qua':
                label = [[float(objects[0].split()[0]), float(objects[0].split()[1]), float(objects[0].split()[2]),
                          float(objects[0].split()[3])],
                         [float(objects[0].split()[4]), float(objects[0].split()[5]), float(objects[0].split()[6])]]
            elif pose_type[:3] == 'deg':
                label = [[float(objects[0].split()[0]), float(objects[0].split()[1]), float(objects[0].split()[2])],
                             [float(objects[0].split()[3])],
                             [float(objects[0].split()[4]), float(objects[0].split()[5]), float(objects[0].split()[6])]]
            testlist.append([vid, first_img, second_img, label])
        # Saving the objects:
        with open(os.path.join(end_path,label_folder+'_list.pkl'), 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([trainlist, testlist, video_list], f)
    else:
        # Getting back the objects:
        with open(os.path.join(end_path,label_folder+'_list.pkl'), 'rb') as f:  # Python 3: open(..., 'rb')
            trainlist, testlist, video_list = pickle.load(f)
    return trainlist, testlist, video_list


class PoseDetection(data.Dataset):
    """JAAD Action Detection Dataset
    to access input images and target which is annotation
    """

    def __init__(self, root, image_set, dataset_name='3dpw', target_type='non_inv', pose_type='quaternion', seven_scene_opt='total', data=None):
        self.root = root
        self.image_set = image_set
        self.dataset_name = dataset_name
        self.target_type = target_type
        self.pose_type = pose_type
        self.seven_scene_opt = seven_scene_opt
        self.data_path = os.path.join(self.root, self.dataset_name)
        self.data = data
        self.ids = list()

        trainlist, testlist, video_list = pose_make_lists(dataset_path=self.root, dataset_name=self.dataset_name, target_type=self.target_type,
                                                                        pose_type=self.pose_type,seven_scene_opt=self.seven_scene_opt)

        self.video_list = video_list
        if self.image_set == 'train':
            self.ids = trainlist
        elif self.image_set == 'test':
            self.ids = testlist
        else:
            print('spacify correct subset ')

    def __getitem__(self, index):
        im1, im2, gt, img_index = self.pull_item(index)

        return im1, im2, gt, img_index

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_path1 = self.ids[index][1]
        img_path2 = self.ids[index][2]
        label_form = self.ids[index][3]
        if self.data == None:
            if self.target_type[:3] == 'non':
                img1 = cv2.imread(os.path.join(self.data_path, img_path1))
                img2 = cv2.imread(os.path.join(self.data_path, img_path2))
            else:
                img2 = cv2.imread(os.path.join(self.data_path, img_path1))
                img1 = cv2.imread(os.path.join(self.data_path, img_path2))

            img1[:, :, :3] = img1[:, :, (2, 1, 0)]
            img2[:, :, :3] = img2[:, :, (2, 1, 0)]
            img1 = cv2.resize(img1, dsize=(224, 224), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, dsize=(224, 224), interpolation=cv2.INTER_AREA)

            img1 = ((img1 / 255.0) - [0.406, 0.456, 0.485]) / [0.229, 0.224, 0.225]
            img2 = ((img2 / 255.0) - [0.406, 0.456, 0.485]) / [0.229, 0.224, 0.225]

            img1 = torch.from_numpy(img1).permute(2, 0, 1).type(torch.FloatTensor)
            img2 = torch.from_numpy(img2).permute(2, 0, 1).type(torch.FloatTensor)
        else:
            if self.target_type[:3] == 'non':
                img1 = self.data[img_path1]
                img2 = self.data[img_path2]
            else:
                img2 = self.data[img_path1]
                img1 = self.data[img_path2]
        return img1,img2, label_form, index
        # return torch.from_numpy(img), target, height, width


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    label_len = len(batch[0][2])
    target1 = []
    target2 = []
    if label_len == 3:
        target3 = []
    imgs1 = []
    imgs2 = []
    image_ids = []
    for sample in batch:
        imgs1.append(sample[0])
        imgs2.append(sample[1])
        target1.append(torch.FloatTensor(sample[2][0]))
        target2.append(torch.FloatTensor(sample[2][1]))
        image_ids.append(sample[3])
        if label_len == 3:
            target3.append(torch.FloatTensor(sample[2][2]))
    if label_len == 3:
        return [torch.stack(imgs1, 0), torch.stack(imgs2, 0), torch.stack(target1, 0), torch.stack(target2, 0),
                torch.stack(target3, 0), image_ids]
    else:
        return [torch.stack(imgs1, 0), torch.stack(imgs2, 0), torch.stack(target1, 0), torch.stack(target2, 0), image_ids]

"""UCF24 Dataset Classes

Author: Gurkirt Singh for ucf101-24 dataset

"""

import os
import os.path
import torch
import torch.utils.data as data
import cv2, pickle
import numpy as np
import os.path as osp


'''CLASSES = (  # always index 0
        'Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving', 'Fencing',
        'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing',
        'SalsaSpin','SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling',
        'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog')'''
#CLASSES = ('non_crossing', 'crossing')
#CLASSES = ('clear path', 'crossing', 'handwave', 'looking', 'nod', 'slow down', 'speed up', 'standing', 'walking', 'background')
CLASSES = ('clear path', 'handwave', 'looking', 'nod', 'slow down', 'speed up', 'standing', 'walking', 'background')
JAAD_ROOT = '/data/jaad/'

class AnnotationTransform(object):
    """
    Same as original
    Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of UCF24's 24 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(CLASSES, range(len(CLASSES))))
        self.ind_to_class = dict(zip(range(len(CLASSES)),CLASSES))

    def __call__(self, labels, width, height):
        res = []
        for t in range(len(labels)):
            label = labels[t]
            '''pts = ['xmin', 'ymin', 'xmax', 'ymax']'''
            bndbox = []
            bndbox.append(label)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


def readsplitfile(splitfile):
    with open(splitfile, 'r') as f:
        temptrainvideos = f.readlines()
    trainvideos = []
    for vid in temptrainvideos:
        vid = vid.rstrip('\n')
        trainvideos.append(vid)
    return trainvideos


def make_lists(rootpath, imgtype, split=1, fulltest=False):
    imagesDir = rootpath + imgtype + '/'
    splitfile = rootpath + 'splitfiles/trainlist{:02d}.txt'.format(split)
    trainvideos = readsplitfile(splitfile)
    trainlist = []
    testlist = []

    with open(rootpath + 'splitfiles/pyannot.pkl','rb') as fff:
        database = pickle.load(fff)

    train_action_counts = np.zeros(len(CLASSES), dtype=np.int32)
    test_action_counts = np.zeros(len(CLASSES), dtype=np.int32)

    #4500ratios = np.asarray([1.1, 0.8, 4.7, 1.4, 0.9, 2.6, 2.2, 3.0, 3.0, 5.0, 6.2, 2.7,
    #                     3.5, 3.1, 4.3, 2.5, 4.5, 3.4, 6.7, 3.6, 1.6, 3.4, 0.6, 4.3])
    ratios = np.asarray([1.03, 0.75, 4.22, 1.32, 0.8, 2.36, 1.99, 2.66, 2.68, 4.51, 5.56, 2.46, 3.17, 2.76, 3.89, 2.28, 4.01, 3.08, 6.06, 3.28, 1.51, 3.05, 0.6, 3.84])
    #ratios = np.ones_like(ratios) #TODO:uncomment this line and line 155, 156 to compute new ratios might be useful for JHMDB21
    video_list = []
    for vid, videoname in enumerate(sorted(database.keys())):
        video_list.append(videoname)
        actidx = database[videoname]['label']
        istrain = True
        step = ratios[actidx]
        numf = database[videoname]['numf']
        lastf = numf-1
        if videoname not in trainvideos:
            istrain = False
            step = max(1, ratios[actidx])*3
        if fulltest:
            step = 1
            lastf = numf

        annotations = database[videoname]['annotations']
        num_tubes = len(annotations)

        tube_labels = np.zeros((numf,num_tubes),dtype=np.int16) # check for each tube if present in
        tube_boxes = [[[] for _ in range(num_tubes)] for _ in range(numf)]
        for tubeid, tube in enumerate(annotations):
            # print('numf00', numf, tube['sf'], tube['ef'])
            for frame_id, frame_num in enumerate(np.arange(tube['sf'], tube['ef'], 1)): # start of the tube to end frame of the tube
                label = tube['label']
                assert actidx == label, 'Tube label and video label should be same'
                box = tube['boxes'][frame_id, :]  # get the box as an array
                box = box.astype(np.float32)
                box[2] += box[0]  #convert width to xmax
                box[3] += box[1]  #converst height to ymax
                tube_labels[frame_num, tubeid] = 1 #label+1  # change label in tube_labels matrix to 1 form 0
                tube_boxes[frame_num][tubeid] = box  # put the box in matrix of lists

        possible_frame_nums = np.arange(0, lastf, step)
        # print('numf',numf,possible_frame_nums[-1])
        for frame_num in possible_frame_nums: # loop from start to last possible frame which can make a legit sequence
            frame_num = int(frame_num)
            check_tubes = tube_labels[frame_num,:]

            if np.sum(check_tubes)>0:  # check if there aren't any semi overlapping tubes
                all_boxes = []
                labels = []
                image_name = imagesDir + videoname+'/{:05d}.jpg'.format(frame_num+1)
                #label_name = rootpath + 'labels/' + videoname + '/{:05d}.txt'.format(frame_num + 1)
                assert os.path.isfile(image_name), 'Image does not exist'+image_name
                for tubeid, tube in enumerate(annotations):
                    label = tube['label']
                    if tube_labels[frame_num, tubeid]>0:
                        box = np.asarray(tube_boxes[frame_num][tubeid])
                        all_boxes.append(box)
                        labels.append(label)

                if istrain: # if it is training video
                    trainlist.append([vid, frame_num+1, np.asarray(labels), np.asarray(all_boxes)])
                    train_action_counts[actidx] += 1 #len(labels)
                else: # if test video and has micro-tubes with GT
                    testlist.append([vid, frame_num+1, np.asarray(labels), np.asarray(all_boxes)])
                    test_action_counts[actidx] += 1 #len(labels)
            elif fulltest and not istrain: # if test video with no ground truth and fulltest is trues
                testlist.append([vid, frame_num+1, np.asarray([9999]), np.zeros((1,4))])

    for actidx, act_count in enumerate(train_action_counts): # just to see the distribution of train and test sets
        print('train {:05d} test {:05d} action {:02d} {:s}'.format(act_count, test_action_counts[actidx] , int(actidx), CLASSES[actidx]))

    newratios = train_action_counts/5000
    #print('new   ratios', newratios)
    line = '['
    for r in newratios:
        line +='{:0.2f}, '.format(r)
    print(line+']')
    print('Trainlistlen', len(trainlist), ' testlist ', len(testlist))

    return trainlist, testlist, video_list


class UCF24Detection(data.Dataset):
    """UCF24 Action Detection Dataset
    to access input images and target which is annotation
    """

    def __init__(self, root, image_set, transform=None, target_transform=None,
                 dataset_name='ucf24', input_type='rgb', full_test=False):

        self.input_type = input_type
        input_type = input_type+'-images'
        self.root = root
        self.CLASSES = CLASSES
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join(root, 'labels/', '%s.txt')
        self._imgpath = os.path.join(root, input_type)
        self.ids = list()

        trainlist, testlist, video_list = make_lists(root, input_type, split=1, fulltest=full_test)
        self.video_list = video_list
        if self.image_set == 'train':
            self.ids = trainlist
        elif self.image_set == 'test':
            self.ids = testlist
        else:
            print('spacify correct subset ')

    def __getitem__(self, index):
        im, gt, img_index = self.pull_item(index)

        return im, gt, img_index

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        annot_info = self.ids[index]
        frame_num = annot_info[1]
        video_id = annot_info[0]
        videoname = self.video_list[video_id]
        img_name = self._imgpath + '/{:s}/{:05d}.jpg'.format(videoname, frame_num)
        # print(img_name)
        img = cv2.imread(img_name)
        height, width, channels = img.shape

        target = self.target_transform(annot_info[3], annot_info[2], width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        # print(height, width,target)
        return torch.from_numpy(img).permute(2, 0, 1), target, index
        # return torch.from_numpy(img), target, height, width


def jaad_make_lists(rootpath, imgtype, split=1, fulltest=False, model_len=15):
    list_path = rootpath + 'smartcar/'

    '''# make video_list
    trainlistpth = list_path + 'trainlist{:02d}.txt'.format(split)
    testlistpth = list_path + 'testlist{:02d}.txt'.format(split)
    train_seq = []
    test_seq = []
    video_list = []
    with open(trainlistpth, 'r') as f:
        train_seq = f.readlines()
        video_list = train_seq
    with open(testlistpth, 'r') as f:
        test_seq = f.readlines()
        video_list = video_list + test_seq

    # make trainlist
    trainlist = []
    for i in train_seq:
        label_path = rootpath + 'labels/' + i[:-1]
        labeltxt = os.listdir(label_path)
        labeltxt.sort()
        for j in labeltxt:
            txt_path = label_path + '/' + j
            framenum = int(j[6:10])
            for idx, k in enumerate(video_list):
                if k == i:
                    vid = idx
                    break
            with open(txt_path, 'r') as f:
                objects = f.readlines()
                objectnum = len(objects)
                boxes = np.zeros((objectnum, 4))
                label = np.zeros((objectnum, 9))
                for k in range(objectnum):
                    boxes[k, 0] = objects[k].split(' ')[0]
                    boxes[k, 1] = objects[k].split(' ')[1]
                    boxes[k, 2] = objects[k].split(' ')[2]
                    boxes[k, 3] = objects[k].split(' ')[3]
                    label[k] = objects[k].split(' ')[4:]
            trainlist.append([vid, framenum, label, boxes])

    # make testlist
    testlist = []
    for i in test_seq:
        label_path = rootpath + 'labels/' + i[:-1]
        labeltxt = os.listdir(label_path)
        labeltxt.sort()
        for j in labeltxt:
            txt_path = label_path + '/' + j
            framenum = int(j[6:10])
            for idx, k in enumerate(video_list):
                if k == i:
                    vid = idx
                    break
            with open(txt_path, 'r') as f:
                objects = f.readlines()
                objectnum = len(objects)
                boxes = np.zeros((objectnum, 4))
                label = np.zeros((objectnum, 9))
                for k in range(objectnum):
                    boxes[k, 0] = objects[k].split(' ')[0]
                    boxes[k, 1] = objects[k].split(' ')[1]
                    boxes[k, 2] = objects[k].split(' ')[2]
                    boxes[k, 3] = objects[k].split(' ')[3]
                    label[k] = objects[k].split(' ')[4:]
            testlist.append([vid, framenum, label, boxes])
    # Saving the objects:
    with open(list_path+'list.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([trainlist, testlist, video_list], f)'''

    # Getting back the objects:
    with open(list_path+'seq_pickle_%02d_half_expand.pkl'%(model_len), 'rb') as f:  # Python 3: open(..., 'rb')
        trainlist, testlist, video_list = pickle.load(f)
    with open(list_path + 'answer_sheet.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        answer_sheet = pickle.load(f)
    return trainlist, testlist, video_list, answer_sheet


class JAADDetection(data.Dataset):
    """JAAD Action Detection Dataset
    to access input images and target which is annotation
    """

    def __init__(self, root, image_set, transform=None, target_transform=None,
                 dataset_name='jaad', input_type='rgb', full_test=False, model_len=15, data=None):

        self.input_type = input_type
        input_type = "smartcar/expand_image"#input_type+'-images'
        self.root = root
        self.CLASSES = CLASSES
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join(root, 'labels/', '%s.txt')
        self._imgpath = os.path.join(root, input_type)
        self.ids = list()
        self.model_len = model_len

        trainlist, testlist, video_list, answer_sheet = jaad_make_lists(root, input_type, split=1, fulltest=full_test, model_len=self.model_len)

        self.img_list = data

        self.video_list = video_list
        if self.image_set == 'train':
            self.ids = trainlist
            self.answer = answer_sheet
        elif self.image_set == 'test':
            self.ids = testlist
            self.answer = answer_sheet
        else:
            print('spacify correct subset ')

    def __getitem__(self, index):
        im, gt, img_index = self.pull_item(index)

        return im, gt, img_index

    def __len__(self):
        return len(self.ids)

    '''def pull_item(self, index):
        annot_info = self.ids[index]
        frame_num = annot_info[1]
        video_id = annot_info[0]
        videoname = self.video_list[video_id][:-1]
        img_name = self._imgpath + '/{:s}/frame_{:04d}.png'.format(videoname, frame_num)
        traffic_name = 'F:\\realtime\\data_dir\\datasets\\jaad\\JAAD_context_num_label\\{:s}\\label_{:04d}.txt'.format(videoname, frame_num)
        with open(traffic_name, "r") as f:
            line = f.readlines()
        traffic_gt = np.zeros(8)
        for idx, i in enumerate(line[0].split(',')):
            traffic_gt[idx] = int(i)
        # print(img_name)
        img = cv2.imread(img_name)
        height, width, channels = img.shape

        target = self.target_transform(annot_info[3], annot_info[2], width, height)

        if self.transform is not None:
            label = []
            targets = []
            for i in range(len(target)):
                label.append(target[i][4:])
                targets.append(target[i][:4])
            target = np.array(targets)
            img, boxes, labels = self.transform(img, target, label)
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            labels = np.squeeze(np.array(labels), axis=1)
            cross = labels[:,1].reshape((-1,1))
            act = np.delete(labels,1,1)
            target = np.hstack((boxes, act, cross))
        # print(height, width,target)
        return torch.from_numpy(img).permute(2, 0, 1), target, index, traffic_gt
        # return torch.from_numpy(img), target, height, width'''
    def pull_item(self, index):
        image = []
        label_out = []
        for i in range(self.model_len):
            annot_info = self.answer[self.ids[index][0] + i]
            frame_num = annot_info[1]
            video_id = annot_info[0]
            videoname = "/".join(self.video_list[video_id][:-1].split('\\')[2:4])

            img_name = self._imgpath + '/{:s}/frame_{:04d}.png'.format(videoname, frame_num)
            #img = cv2.imread(img_name)
            img = self.img_list[img_name]
            height, width, channels = img.shape

            target = self.target_transform(annot_info[2], width, height)

            if self.transform is not None:
                label = target[0][0]
                img, labels = self.transform(img, label)
                img[:,:,:3] = img[:, :, (2, 1, 0)]
                cross = labels[-1].reshape((-1,1))
                act = labels[:8].reshape((-1,8))
                traffic_gt = labels[8:-1].reshape((-1,6))
                target = np.hstack((act, traffic_gt, cross))
            image.append(torch.from_numpy(img).permute(2, 0, 1))
            label_out.append(target)
        return image, label_out, index
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

    targets = []
    imgs = []
    image_ids = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        image_ids.append(sample[2])
    return torch.stack(imgs, 0), targets, image_ids

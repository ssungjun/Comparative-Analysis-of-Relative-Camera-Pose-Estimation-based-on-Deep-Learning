import numpy as np
import pickle
from tqdm import tqdm
import os
import cv2
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def quaternion_to_rotation(quaternion):
    quaternion_cpu = np.array(quaternion.data.cpu())
    scaled_quaternion = np.zeros(quaternion_cpu.shape)
    batch_size = quaternion_cpu.shape[0]
    rotation_matrix = np.zeros((batch_size, 3, 3))
    for i in range(batch_size):
        temp_scale = np.sqrt(np.power(quaternion_cpu[i][0], 2) + np.power(quaternion_cpu[i][1], 2)
                             + np.power(quaternion_cpu[i][2], 2) + np.power(quaternion_cpu[i][3], 2))
        scaled_quaternion[i] = quaternion_cpu[i] / temp_scale
        rotation_matrix[i][0][0] = 1 - 2 * np.power(scaled_quaternion[i][2],2) - 2 * np.power(scaled_quaternion[i][3],
                                                                                              2)
        rotation_matrix[i][0][1] = 2 * (scaled_quaternion[i][1] * scaled_quaternion[i][2]
                                        - scaled_quaternion[i][3] * scaled_quaternion[i][0])
        rotation_matrix[i][0][2] = 2 * (scaled_quaternion[i][1] * scaled_quaternion[i][3]
                                        + scaled_quaternion[i][2] * scaled_quaternion[i][0])
        rotation_matrix[i][1][0] = 2 * (scaled_quaternion[i][1] * scaled_quaternion[i][2]
                                        + scaled_quaternion[i][3] * scaled_quaternion[i][0])
        rotation_matrix[i][1][1] = 1 - 2 * np.power(scaled_quaternion[i][1], 2) - 2 * np.power(scaled_quaternion[i][3],
                                                                                               2)
        rotation_matrix[i][1][2] = 2 * (scaled_quaternion[i][2] * scaled_quaternion[i][3]
                                        - scaled_quaternion[i][1] * scaled_quaternion[i][0])
        rotation_matrix[i][2][0] = 2 * (scaled_quaternion[i][1] * scaled_quaternion[i][3]
                                        - scaled_quaternion[i][2] * scaled_quaternion[i][0])
        rotation_matrix[i][2][1] = 2 * (scaled_quaternion[i][2] * scaled_quaternion[i][3]
                                        + scaled_quaternion[i][1] * scaled_quaternion[i][0])
        rotation_matrix[i][2][2] = 1 - 2 * np.power(scaled_quaternion[i][1], 2) - 2 * np.power(scaled_quaternion[i][2],
                                                                                               2)
    return rotation_matrix


def rotation_error(rot1, rot2):
    batch_size = rot1.shape[0]
    angle_error = []
    for i in range(batch_size):
        r_ab = rot1[i].T * rot2[i]
        angle_error.append(np.rad2deg(np.arccos((np.trace(r_ab) - 1) / 2)))
    mean_of_angle_error = np.mean(angle_error)
    return mean_of_angle_error


def pre_read_data(arg):
    mass = 40
    img_list = {}
    img_path = os.path.join(arg.dataset_root, arg.dataset)
    if (os.path.isfile(os.path.join(img_path, 'datalist', 'data_%02d.pkl'%(mass - 1)))) and ((arg.dataset[0] != '7') or (arg.seven_opt[0] == 't')):
        for i in range(mass):
            with open(os.path.join(img_path, 'datalist', 'data_%02d.pkl' % (i)), 'rb') as f:
                temp = pickle.load(f)
                img_list.update(temp)
    elif (os.path.isfile(os.path.join(img_path, arg.seven_opt, 'datalist', 'data_%02d.pkl'%(mass - 1)))) and (arg.dataset[0] == '7'):
        for i in range(mass):
            with open(os.path.join(img_path, 'datalist', 'data_%02d.pkl' % (i)), 'rb') as f:
                temp = pickle.load(f)
                img_list.update(temp)
    else:
        if not arg.dataset[0] == '7':
            with open(os.path.join(img_path, 'Total_file_list.txt'), "r") as f:
                temp_list = f.readlines()
        else:
            if arg.seven_opt[0] == 't':
                with open(os.path.join(img_path, 'Total_file_list.txt'), "r") as f:
                    temp_list = f.readlines()
            else:
                with open(os.path.join(img_path, arg.seven_opt, 'Total_file_list.txt'), "r") as f:
                    temp_list = f.readlines()

        if not arg.dataset[0] == '7':
            list_path = os.path.join(arg.dataset_root, arg.dataset, 'datalist')
        else:
            if arg.seven_opt[0] == 't':
                list_path = os.path.join(arg.dataset_root, arg.dataset, 'datalist')
            else:
                list_path = os.path.join(arg.dataset_root, arg.dataset, arg.seven_opt, 'datalist')
        if not os.path.isdir(list_path):
            os.makedirs(list_path)
        data_len = len(temp_list)
        save_idx = 0
        temp_dict = {}
        count = 0
        for i in tqdm(temp_list, desc='image_read', mininterval=1):
            img2 = cv2.imread(os.path.join(img_path, i[:-1]))
            img2[:, :, :3] = img2[:, :, (2, 1, 0)]
            img2 = cv2.resize(img2, dsize=(224, 224), interpolation=cv2.INTER_AREA)
            img2 = ((img2 / 255.0) - [0.406, 0.456, 0.485]) / [0.229, 0.224, 0.225]
            img2 = torch.from_numpy(img2).permute(2, 0, 1).type(torch.FloatTensor)
            img_list[i[:-1]] = img2
            temp_dict[i[:-1]] = img2
            if count == (round((data_len * (save_idx + 1)) / mass) - 1):
                with open(os.path.join(list_path, 'data_%02d.pkl' % (save_idx)), 'wb') as f:
                    pickle.dump(temp_dict, f)
                save_idx = save_idx + 1
                print(len(temp_dict))
                temp_dict = {}
            count = count + 1
    return img_list


def quaternion_translation_to_transform(quaternion, trans):
    transformaton_matrix = np.zeros((4,4))
    reshape_quat = np.array(quaternion).reshape((1,-1))
    rot_mat = quaternion_to_rotation(torch.from_numpy(reshape_quat))
    transformaton_matrix[:3,:3] = rot_mat[0]
    transformaton_matrix[:3,3] = np.transpose(np.array(trans))
    transformaton_matrix[3, 3] = 1
    return transformaton_matrix


def quaternion_error(quat_out, quat_target):
    batch_size = quat_out.shape[0]
    angle_error = []
    for i in range(batch_size):
        norm_quat_p1 = torch.sqrt(torch.sum(quat_out[i] ** 2))
        scaled_quat_p1 = torch.div(quat_out[i].T, norm_quat_p1).T
        norm_quat_p2 = torch.sqrt(torch.sum(quat_target[i] ** 2))
        scaled_quat_p2 = torch.div(quat_target[i].T, norm_quat_p2).T
        r_ab = torch.sum(scaled_quat_p1 * scaled_quat_p2).cpu().data
        angle_error.append(np.rad2deg(np.arccos(r_ab))*2)
    mean_of_angle_error = np.mean(angle_error)
    return mean_of_angle_error

def diff_quaternion(quaternion, args):
    if args.dataset[0] == 'k':
        fps = 10
    else:
        fps = 30
    diff_quat = quaternion.clone()
    batch_size = quaternion.shape[0]
    for i in range(batch_size):
        sin_theta = torch.sqrt(quaternion[i, 1]**2 + quaternion[i, 2]**2 + quaternion[i, 3]**2)
        rotation_bar = [quaternion[i, 1]/sin_theta, quaternion[i, 2]/sin_theta, quaternion[i, 3]/sin_theta]
        velocity_bar = rotation_bar
        half_theta = torch.atan2(sin_theta, quaternion[i, 0])
        angular_velocity = half_theta * 2 * fps
        velocity_bar[0] *= angular_velocity
        velocity_bar[1] *= angular_velocity
        velocity_bar[2] *= angular_velocity
        diff_quat[i, 0] = (-velocity_bar[0] * quaternion[i, 1] - velocity_bar[1] * quaternion[i, 2] - velocity_bar[2] *
                           quaternion[i, 3]) / 2
        diff_quat[i, 1] = (velocity_bar[0] * quaternion[i, 0] + velocity_bar[2] * quaternion[i, 2] - velocity_bar[1] *
                           quaternion[i, 3]) / 2
        diff_quat[i, 2] = (velocity_bar[1] * quaternion[i, 0] - velocity_bar[2] * quaternion[i, 1] + velocity_bar[0] *
                           quaternion[i, 3]) / 2
        diff_quat[i, 3] = (velocity_bar[2] * quaternion[i, 0] + velocity_bar[1] * quaternion[i, 1] - velocity_bar[0] *
                           quaternion[i, 2]) / 2
    return diff_quat


def diff_transport(trans, args):
    if args.dataset[0] == 'k':
        fps = 10
    else:
        fps = 30
    diff_transport = trans.clone() * fps
    return diff_transport


def diff_quaternion_simple(quaternion, args):
    if args.dataset[0] == 'k':
        fps = 10
    else:
        fps = 30
    diff_quat = quaternion.clone()
    batch_size = quaternion.shape[0]
    for i in range(batch_size):
        sin_theta = torch.sqrt(quaternion[i, 1]**2 + quaternion[i, 2]**2 + quaternion[i, 3]**2)
        cos_theta = torch.sqrt(1.0 - (sin_theta**2))
        rotation_bar = [quaternion[i, 1]/sin_theta, quaternion[i, 2]/sin_theta, quaternion[i, 3]/sin_theta]
        half_theta = torch.atan2(sin_theta, quaternion[i, 0])
        angular_velocity = half_theta * 2 * fps
        diff_quat[i, 0] = (-sin_theta * angular_velocity) / 2
        diff_quat[i, 1] = (rotation_bar[0] * cos_theta * angular_velocity) / 2
        diff_quat[i, 2] = (rotation_bar[1] * cos_theta * angular_velocity) / 2
        diff_quat[i, 3] = (rotation_bar[2] * cos_theta * angular_velocity) / 2

    return diff_quat


def diff_quaternion_to_quat(quaternion, args):
    if args.dataset[0] == 'k':
        fps = 10
    else:
        fps = 30
    inte_quat = quaternion.clone()
    batch_size = quaternion.shape[0]
    for i in range(batch_size):
        angular_velocity = torch.sqrt(
            quaternion[i, 1] ** 2 + quaternion[i, 2] ** 2 + quaternion[i, 3] ** 2 + quaternion[i, 0] ** 2) * 2
        sin_theta = (quaternion[i, 0] * -2.0) / angular_velocity
        cos_theta = torch.sqrt(1.0 - (sin_theta ** 2))
        rotation_bar = [(quaternion[i, 1] * 2) / (cos_theta * angular_velocity),
                        (quaternion[i, 2] * 2) / (cos_theta * angular_velocity),
                        (quaternion[i, 3] * 2) / (cos_theta * angular_velocity)]
        inte_quat[i, 0] = cos_theta
        inte_quat[i, 1] = rotation_bar[0] * sin_theta
        inte_quat[i, 2] = rotation_bar[1] * sin_theta
        inte_quat[i, 3] = rotation_bar[2] * sin_theta
    return inte_quat


def accumulated_quaternion(quat_first, quat_second):
    batch_size = quat_first.shape[0]
    accu_quat = torch.zeros(quat_first.shape)#quat_first.clone()
    for i in range(batch_size):
        accu_quat[i, 0] = torch.add(torch.add(torch.mul(quat_first[i, 0], quat_second[i, 0]), torch.mul(quat_first[i, 1], quat_second[i, 1]), alpha=-1),
                                torch.add(torch.mul(quat_first[i, 2], quat_second[i, 2]), torch.mul(quat_first[i, 3], quat_second[i, 3])), alpha=-1)
        accu_quat[i, 1] = torch.add(torch.add(torch.mul(quat_first[i, 0], quat_second[i, 1]), torch.mul(quat_first[i, 1], quat_second[i, 0])),
                                torch.add(torch.mul(quat_first[i, 2], quat_second[i, 3]), torch.mul(quat_first[i, 3], quat_second[i, 2]), alpha=-1))
        accu_quat[i, 2] = torch.add(torch.add(torch.mul(quat_first[i, 0], quat_second[i, 2]), torch.mul(quat_first[i, 1], quat_second[i, 3]), alpha=-1),
                                torch.add(torch.mul(quat_first[i, 2], quat_second[i, 0]), torch.mul(quat_first[i, 3], quat_second[i, 1])))
        accu_quat[i, 3] = torch.add(torch.add(torch.mul(quat_first[i, 0], quat_second[i, 3]), torch.mul(quat_first[i, 1], quat_second[i, 2])),
                                torch.add(torch.mul(quat_first[i, 2], quat_second[i, 1]), torch.mul(quat_first[i, 3], quat_second[i, 0]), alpha=-1), alpha=-1)
        norm_quat = torch.sqrt(torch.sum(accu_quat[i] ** 2))
        accu_quat[i] = torch.div(accu_quat[i].T, norm_quat).T
    return accu_quat
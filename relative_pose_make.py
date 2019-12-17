# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os
import shutil


def inverse_transform(mat):
    temp_mat = mat.transpose()
    inverse_mat = np.zeros([4, 4])
    inv_rot = np.transpose(temp_mat[:-1,:-1])
    trans = temp_mat[:-1,-1].reshape((1,3)).transpose()
    inv_tran = np.reshape(-np.matmul(inv_rot,trans), (3))
    inverse_mat[:-1,:-1] = inv_rot
    inverse_mat[:-1, -1] = inv_tran
    inverse_mat[-1,-1] = 1
    return inverse_mat.transpose()


def transform_to_quaternion(mat):
    rot_mat = mat[:-1,:-1]
    q0 = np.sqrt(1 + rot_mat[0, 0] + rot_mat[1, 1] + rot_mat[2, 2]) / 2
    q1 = (rot_mat[1, 2] - rot_mat[2, 1]) / (4 * q0)
    q2 = (rot_mat[2, 0] - rot_mat[0, 2]) / (4 * q0)
    q3 = (rot_mat[0, 1] - rot_mat[1, 0]) / (4 * q0)
    return [q0, q1, q2, q3], [mat[-1, 0], mat[-1, 1], mat[-1, 2]]


def quaternion_to_vec_degree(quat):
    sin = np.sqrt(np.power(quat[1], 2) + np.power(quat[2], 2) + np.power(quat[3], 2))
    kx = quat[1] / sin
    ky = quat[2] / sin
    kz = quat[3] / sin
    degree = np.arctan2(sin, quat[0]) * 180 / np.pi
    return [kx, ky, kz], degree
'''##############################################3DPW######################################################
#####################GLOBAL#######################
base_path = "G:\\dataset\\3dpw\\sequenceFiles\\"
save_path = "G:\\dataset\\3dpw\\global_pose\\"

ttv = os.listdir(base_path)
ttv.sort()
for directory in ttv:
    fulldirname = os.path.join(base_path, directory)
    files = os.listdir(fulldirname)
    files.sort()
    for file in files:
        fullfilename = os.path.join(fulldirname, file)
        data_list = []
        with open(fullfilename, 'rb') as file:
             while True:
                 try:
                     data = pickle.load(file, encoding='latin1')
                 except EOFError:
                     break
                 data_list.append(data)
        global_poses = data_list[0]["cam_poses"]

        if not os.path.isdir(os.path.join(save_path,file.name.split('\\')[-1][:-4])):
            os.makedirs(os.path.join(save_path,file.name.split('\\')[-1][:-4]))

        for i in range(len(global_poses)):
            global_pose = global_poses[i,:,:]
            with open(os.path.join(save_path,file.name.split('\\')[-1][:-4],"frame_%05d.txt"%(i)), 'wb') as save_file:
                for x in range(4):
                    for y in range(4):
                        para = global_pose[x,y]
                        save_file.write("%f    ".encode()%(para))
                    save_file.write('\n'.encode())
##################################################GLOBAL END#################################
###################RELATIVE######################
base_path = "G:\\dataset\\3dpw\\global_pose\\"
save_path = "G:\\dataset\\3dpw\\relative_pose\\"

seq = os.listdir(base_path)
seq.sort()
for directory in seq:
    fulldirname = os.path.join(base_path, directory)
    files = os.listdir(fulldirname)
    files.sort()
    for index in range(len(files[:-1])):
        previous_name = os.path.join(fulldirname, "frame_%05d.txt" % (index))
        present_name = os.path.join(fulldirname, "frame_%05d.txt" % (index+1))
        data_list = []
        with open(previous_name, 'rb') as file:
            prev_data = file.readlines()
        with open(present_name, 'rb') as file:
            pres_data = file.readlines()
        prev_transform = np.zeros([4, 4])
        pres_transform = np.zeros([4, 4])
        for x in range(4):
            for y in range(4):
                prev_transform[y, x] = float(prev_data[x].split()[y])
                pres_transform[y, x] = float(pres_data[x].split()[y])
        relative_transform = np.matmul(inverse_transform(prev_transform).transpose(), pres_transform.transpose()).transpose()
        if not os.path.isdir(os.path.join(save_path,file.name.split('\\')[-2])):
            os.makedirs(os.path.join(save_path,file.name.split('\\')[-2]))

        with open(os.path.join(save_path,file.name.split('\\')[-2],"frame_%05d.txt"%(index)), 'wb') as save_file:
            for x in range(4):
                for y in range(4):
                    para = relative_transform[y,x]
                    save_file.write("%f    ".encode()%(para))
                save_file.write('\n'.encode())
###################INVERSE RELATIVE######################
base_path = "G:\\dataset\\3dpw\\relative_pose\\"
save_path = "G:\\dataset\\3dpw\\inverse_relative_pose\\"

seq = os.listdir(base_path)
seq.sort()
for directory in seq:
    fulldirname = os.path.join(base_path, directory)
    files = os.listdir(fulldirname)
    files.sort()
    for index in range(len(files)):
        relative_name = os.path.join(fulldirname, "frame_%05d.txt" % (index))
        with open(relative_name, 'rb') as file:
            relative_data = file.readlines()
        relative_transform = np.zeros([4, 4])

        for x in range(4):
            for y in range(4):
                relative_transform[y, x] = float(relative_data[x].split()[y])

        inv_relative_transform = inverse_transform(relative_transform)
        if not os.path.isdir(os.path.join(save_path,directory)):
            os.makedirs(os.path.join(save_path,directory))

        with open(os.path.join(save_path,directory,"frame_%05d.txt"%(index)), 'wb') as save_file:
            for x in range(4):
                for y in range(4):
                    para = inv_relative_transform[y,x]
                    save_file.write("%f    ".encode()%(para))
                save_file.write('\n'.encode())
###################QUATERNION######################
base_path = "G:\\dataset\\3dpw\\relative_pose\\"
save_quat_path = "G:\\dataset\\3dpw\\relative_quaternion_pose\\"
save_inv_quat_path = "G:\\dataset\\3dpw\\inverse_relative_quaternion_pose\\"
save_deg_path = "G:\\dataset\\3dpw\\relative_degree_pose\\"
save_inv_deg_path = "G:\\dataset\\3dpw\\inverse_relative_degree_pose\\"

seq = os.listdir(base_path)
seq.sort()
for directory in seq:
    fulldirname = os.path.join(base_path, directory)
    files = os.listdir(fulldirname)
    files.sort()
    for index in range(len(files)):
        relative_name = os.path.join(fulldirname, "frame_%05d.txt" % (index))
        with open(relative_name, 'rb') as file:
            relative_data = file.readlines()
        relative_transform = np.zeros([4, 4])

        for x in range(4):
            for y in range(4):
                relative_transform[y, x] = float(relative_data[x].split()[y])

        inv_relative_transform = inverse_transform(relative_transform)
        quat, translate = transform_to_quaternion(relative_transform)
        inv_quat, inv_trans = transform_to_quaternion(inv_relative_transform)
        vec, degree = quaternion_to_vec_degree(quat)
        inv_vec, inv_degree = quaternion_to_vec_degree(inv_quat)

        if not os.path.isdir(os.path.join(save_quat_path,directory)):
            os.makedirs(os.path.join(save_quat_path,directory))
        if not os.path.isdir(os.path.join(save_inv_quat_path,directory)):
            os.makedirs(os.path.join(save_inv_quat_path,directory))
        if not os.path.isdir(os.path.join(save_deg_path,directory)):
            os.makedirs(os.path.join(save_deg_path,directory))
        if not os.path.isdir(os.path.join(save_inv_deg_path,directory)):
            os.makedirs(os.path.join(save_inv_deg_path,directory))

        with open(os.path.join(save_quat_path,directory,"frame_%05d.txt"%(index)), 'wb') as save_file:
                para = quat + translate
                for term in para:
                    save_file.write("%f    ".encode()%(term))
        with open(os.path.join(save_inv_quat_path,directory,"frame_%05d.txt"%(index)), 'wb') as save_file:
                para = inv_quat + inv_trans
                for term in para:
                    save_file.write("%f    ".encode()%(term))
        with open(os.path.join(save_deg_path,directory,"frame_%05d.txt"%(index)), 'wb') as save_file:
                para = vec + [degree] + translate
                for term in para:
                    save_file.write("%f    ".encode()%(term))
        with open(os.path.join(save_inv_deg_path,directory,"frame_%05d.txt"%(index)), 'wb') as save_file:
                para = inv_vec + [inv_degree] + inv_trans
                for term in para:
                    save_file.write("%f    ".encode()%(term))
#########################7SCENES############################'''
'''##################moving image, depth, pose##################################
base_path = "G:\\dataset\\7scenes\\"
situations = os.listdir(base_path)
situations.sort()
for situation in situations:
    full_situation = os.path.join(base_path, situation)
    seqs = os.listdir(full_situation)
    seqs.sort()
    for seq in seqs:
        if not seq[:3] == 'seq':
            continue
        base_seq = os.path.join(base_path, situation, seq)
        seq_name = seq
        table = str.maketrans('-', '_')
        seq_name = seq_name.translate(table)
        image_seq = os.path.join(base_path, situation, seq_name + "_image")
        depth_seq = os.path.join(base_path, situation, seq_name + "_depth")
        pose_seq = os.path.join(base_path, situation, seq_name + "_pose")
        if not os.path.isdir(image_seq):
            os.makedirs(image_seq)
        if not os.path.isdir(depth_seq):
            os.makedirs(depth_seq)
        if not os.path.isdir(pose_seq):
            os.makedirs(pose_seq)
        files = os.listdir(base_seq)
        files.sort()
        for file in files:
            filepath = os.path.join(base_path, situation, seq, file)
            if file.split('.')[1][:4] == 'colo':
                shutil.move(filepath,image_seq)
            elif file.split('.')[1][:4] == 'dept':
                shutil.move(filepath, depth_seq)
            else:
                shutil.move(filepath, pose_seq)'''
############################RELATIVE POSE#########################
'''base_path = "G:\\dataset\\7scenes\\"

situations = os.listdir(base_path)
situations.sort()
for situation in situations:
    if situation[0] == 'd':
        continue
    situation_path = os.path.join(base_path, situation)
    seqs = os.listdir(situation_path)
    seqs.sort()
    for seq in seqs:
        if not seq[:3] == 'seq':
            continue
        pose_path = os.path.join(base_path, situation, seq, 'pose')
        save_path = os.path.join(base_path, situation, seq, 'relative_pose')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        files = os.listdir(pose_path)
        if 'Thumbs.db' in files:
            del files[files.index('Thumbs.db')]
        files.sort()
        for index in range(len(files) - 1):
            previous_name = os.path.join(pose_path, "frame-%06d.pose.txt" % (index))
            present_name = os.path.join(pose_path, "frame-%06d.pose.txt" % (index + 1))
            data_list = []
            with open(previous_name, 'rb') as file:
                prev_data = file.readlines()
            with open(present_name, 'rb') as file:
                pres_data = file.readlines()
            prev_transform = np.zeros([4, 4])
            pres_transform = np.zeros([4, 4])
            for x in range(4):
                for y in range(4):
                    prev_transform[y, x] = float(prev_data[x].split()[y])
                    pres_transform[y, x] = float(pres_data[x].split()[y])
            relative_transform = np.matmul(inverse_transform(prev_transform).transpose(), pres_transform.transpose()).transpose()

            with open(os.path.join(save_path,"frame-%06d.pose.txt"%(index)), 'wb') as save_file:
                for x in range(4):
                    for y in range(4):
                        para = relative_transform[y,x]
                        save_file.write("%f    ".encode()%(para))
                    save_file.write('\n'.encode())
############################INVERSE RELATIVE POSE#########################
base_path = "G:\\dataset\\7scenes\\"

situations = os.listdir(base_path)
situations.sort()
for situation in situations:
    if situation[0] == 'd':
        continue
    situation_path = os.path.join(base_path, situation)
    seqs = os.listdir(situation_path)
    seqs.sort()
    for seq in seqs:
        if not seq[:3] == 'seq':
            continue
        pose_path = os.path.join(base_path, situation, seq, 'relative_pose')
        save_path = os.path.join(base_path, situation, seq, 'inverse_relative_pose')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        files = os.listdir(pose_path)
        if 'Thumbs.db' in files:
            del files[files.index('Thumbs.db')]
        files.sort()
        for index in range(len(files)):
            relative_name = os.path.join(pose_path, "frame-%06d.pose.txt" % (index))
            with open(relative_name, 'rb') as file:
                relative_data = file.readlines()

            relative_transform = np.zeros([4, 4])
            for x in range(4):
                for y in range(4):
                    relative_transform[y, x] = float(relative_data[x].split()[y])
            inv_relative_transform = inverse_transform(relative_transform)

            with open(os.path.join(save_path,"frame-%06d.pose.txt"%(index)), 'wb') as save_file:
                for x in range(4):
                    for y in range(4):
                        para = inv_relative_transform[y,x]
                        save_file.write("%f    ".encode()%(para))
                    save_file.write('\n'.encode())
############################QUATERNION#########################
base_path = "G:\\dataset\\7scenes\\"

situations = os.listdir(base_path)
situations.sort()
for situation in situations:
    if situation[0] == 'd':
        continue
    situation_path = os.path.join(base_path, situation)
    seqs = os.listdir(situation_path)
    seqs.sort()
    for seq in seqs:
        if not seq[:3] == 'seq':
            continue
        pose_path = os.path.join(base_path, situation, seq, 'relative_pose')
        save_quat_path = os.path.join(base_path, situation, seq, 'relative_quaternion_pose')
        save_inv_quat_path = os.path.join(base_path, situation, seq, 'inverse_relative_quaternion_pose')
        save_deg_path = os.path.join(base_path, situation, seq, 'relative_degree_pose')
        save_inv_deg_path = os.path.join(base_path, situation, seq, 'inverse_relative_degree_pose')
        if not os.path.isdir(save_quat_path):
            os.makedirs(save_quat_path)
        if not os.path.isdir(save_inv_quat_path):
            os.makedirs(save_inv_quat_path)
        if not os.path.isdir(save_deg_path):
            os.makedirs(save_deg_path)
        if not os.path.isdir(save_inv_deg_path):
            os.makedirs(save_inv_deg_path)
        files = os.listdir(pose_path)
        if 'Thumbs.db' in files:
            del files[files.index('Thumbs.db')]
        files.sort()
        for index in range(len(files)):
            relative_name = os.path.join(pose_path, "frame-%06d.pose.txt" % (index))
            with open(relative_name, 'rb') as file:
                relative_data = file.readlines()

            relative_transform = np.zeros([4, 4])
            for x in range(4):
                for y in range(4):
                    relative_transform[y, x] = float(relative_data[x].split()[y])
            inv_relative_transform = inverse_transform(relative_transform)
            quat, translate = transform_to_quaternion(relative_transform)
            inv_quat, inv_trans = transform_to_quaternion(inv_relative_transform)
            vec, degree = quaternion_to_vec_degree(quat)
            inv_vec, inv_degree = quaternion_to_vec_degree(inv_quat)

            with open(os.path.join(save_quat_path,"frame-%06d.pose.txt"%(index)), 'wb') as save_file:
                para = quat + translate
                for term in para:
                    save_file.write("%f    ".encode() % (term))
            with open(os.path.join(save_inv_quat_path,"frame-%06d.pose.txt"%(index)), 'wb') as save_file:
                para = inv_quat + inv_trans
                for term in para:
                    save_file.write("%f    ".encode() % (term))
            with open(os.path.join(save_deg_path,"frame-%06d.pose.txt"%(index)), 'wb') as save_file:
                para = vec + [degree] + translate
                for term in para:
                    save_file.write("%f    ".encode() % (term))
            with open(os.path.join(save_inv_deg_path,"frame-%06d.pose.txt"%(index)), 'wb') as save_file:
                para = inv_vec + [inv_degree] + inv_trans
                for term in para:
                    save_file.write("%f    ".encode() % (term))
###################KITTI ODOMETRY#############################
############RELATIVE POSE######################
base_path = 'G:\\dataset\\kitti\\poses'
save_path = 'G:\\dataset\\kitti\\relative_pose'
seqs = os.listdir(base_path)
seqs.sort()
for seq in seqs:
    with open(os.path.join(base_path,seq), 'rb') as file:
        datas = file.readlines()
    for index in range(len(datas) - 1):
        prev_data = datas[index].split()
        pres_data = datas[index + 1].split()
        prev_transform = np.zeros([4, 4])
        pres_transform = np.zeros([4, 4])
        for i in range(len(prev_data)):
            x = i % 4
            y = int(i / 4)
            prev_transform[x, y] = float(prev_data[i])
            pres_transform[x, y] = float(pres_data[i])
        prev_transform[-1, -1] = 1
        pres_transform[-1, -1] = 1
        relative_transform = np.matmul(inverse_transform(prev_transform).transpose(), pres_transform.transpose()).transpose()
        save_paths = os.path.join(save_path,seq[:-4])
        if not os.path.isdir(save_paths):
            os.makedirs(save_paths)
        with open(os.path.join(save_paths, "%06d.txt" % (index)), 'wb') as save_file:
            for x in range(4):
                for y in range(4):
                    para = relative_transform[y, x]
                    save_file.write("%f    ".encode() % (para))
                save_file.write('\n'.encode())'''
############INVERSE RELATIVE POSE######################
base_path = 'G:\\dataset\\kitti\\relative_pose'
save_path = 'G:\\dataset\\kitti\\inverse_relative_pose'
seqs = os.listdir(base_path)
seqs.sort()
for seq in seqs:
    files_path = os.path.join(base_path,seq)
    files = os.listdir(files_path)
    files.sort()
    for index in range(len(files)):
        with open(os.path.join(files_path,'%06d.txt'%(index)), 'rb') as file:
            relative_data = file.readlines()

        relative_transform = np.zeros([4, 4])
        for x in range(4):
            for y in range(4):
                relative_transform[y, x] = float(relative_data[x].split()[y])
        inv_relative_transform = inverse_transform(relative_transform)
        if not os.path.isdir(os.path.join(save_path, seq)):
            os.makedirs(os.path.join(save_path, seq))
        with open(os.path.join(save_path, seq, "%06d.txt" % (index)), 'wb') as save_file:
            for x in range(4):
                for y in range(4):
                    para = inv_relative_transform[y, x]
                    save_file.write("%f    ".encode() % (para))
                save_file.write('\n'.encode())
############QUATERNION######################
base_path = 'G:\\dataset\\kitti\\relative_pose'
save_quat_path = "G:\\dataset\\kitti\\relative_quaternion_pose\\"
save_inv_quat_path = "G:\\dataset\\kitti\\inverse_relative_quaternion_pose\\"
save_deg_path = "G:\\dataset\\kitti\\relative_degree_pose\\"
save_inv_deg_path = "G:\\dataset\\kitti\\inverse_relative_degree_pose\\"

seqs = os.listdir(base_path)
seqs.sort()
for seq in seqs:
    files_path = os.path.join(base_path,seq)
    files = os.listdir(files_path)
    files.sort()
    for index in range(len(files)):
        with open(os.path.join(files_path,'%06d.txt'%(index)), 'rb') as file:
            relative_data = file.readlines()

        relative_transform = np.zeros([4, 4])
        for x in range(4):
            for y in range(4):
                relative_transform[y, x] = float(relative_data[x].split()[y])

        inv_relative_transform = inverse_transform(relative_transform)
        quat, translate = transform_to_quaternion(relative_transform)
        inv_quat, inv_trans = transform_to_quaternion(inv_relative_transform)
        vec, degree = quaternion_to_vec_degree(quat)
        inv_vec, inv_degree = quaternion_to_vec_degree(inv_quat)

        if not os.path.isdir(os.path.join(save_quat_path, seq)):
            os.makedirs(os.path.join(save_quat_path, seq))
        if not os.path.isdir(os.path.join(save_inv_quat_path, seq)):
            os.makedirs(os.path.join(save_inv_quat_path, seq))
        if not os.path.isdir(os.path.join(save_deg_path, seq)):
            os.makedirs(os.path.join(save_deg_path, seq))
        if not os.path.isdir(os.path.join(save_inv_deg_path, seq)):
            os.makedirs(os.path.join(save_inv_deg_path, seq))

        with open(os.path.join(save_quat_path, seq, "%06d.txt" % (index)), 'wb') as save_file:
            para = quat + translate
            for term in para:
                save_file.write("%f    ".encode() % (term))
        with open(os.path.join(save_inv_quat_path, seq, "%06d.txt" % (index)), 'wb') as save_file:
            para = inv_quat + inv_trans
            for term in para:
                save_file.write("%f    ".encode() % (term))
        with open(os.path.join(save_deg_path, seq, "%06d.txt" % (index)), 'wb') as save_file:
            para = vec + [degree] + translate
            for term in para:
                save_file.write("%f    ".encode() % (term))
        with open(os.path.join(save_inv_deg_path, seq, "%06d.txt" % (index)), 'wb') as save_file:
            para = inv_vec + [inv_degree] + inv_trans
            for term in para:
                save_file.write("%f    ".encode() % (term))

print("end")
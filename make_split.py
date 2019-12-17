# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os
import shutil

length = -4
train_name = 'Train_file_multi_list.txt'
test_name = 'Test_file_multi_list.txt'
######################7SCENES####################
base_path = "G:\\dataset\\7scenes\\"
total_train_list = []
total_test_list = []
situations = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']#os.listdir(base_path)
situations.sort()
for situation in situations:
    seq_path = os.path.join(base_path,situation)
    train_file = os.path.join(seq_path, 'TrainSplit.txt')
    test_file = os.path.join(seq_path, 'TestSplit.txt')

    with open(train_file, 'rb') as file:
        trainseq = file.readlines()
    with open(test_file, 'rb') as file:
        testseq = file.readlines()
    for i in range(len(trainseq)):
        trainseq[i] = int(trainseq[i][8:])
    for i in range(len(testseq)):
        testseq[i] = int(testseq[i][8:])

    train_list = []
    test_list = []

    for seq in trainseq:
        list_path = os.path.join(seq_path, 'seq-%02d'%(seq), 'image')
        files_list = os.listdir(list_path)
        files_list.sort()
        for file in files_list[:length]:
            train_list.append(os.path.join(situation, 'seq-%02d'%(seq), 'image', file))
    for seq in testseq:
        list_path = os.path.join(seq_path, 'seq-%02d'%(seq), 'image')
        files_list = os.listdir(list_path)
        files_list.sort()
        for file in files_list[:length]:
            test_list.append(os.path.join(situation, 'seq-%02d'%(seq), 'image', file))
    total_train_list = total_train_list + train_list
    total_test_list = total_test_list + test_list
    with open(os.path.join(seq_path, train_name), 'wb') as save_file:
        for term in train_list:
            save_file.write((term + '\n').encode())
    with open(os.path.join(seq_path, test_name), 'wb') as save_file:
        for term in test_list:
            save_file.write((term + '\n').encode())
with open(os.path.join(base_path, train_name), 'wb') as save_file:
    for term in total_train_list:
        save_file.write((term + '\n').encode())
with open(os.path.join(base_path, test_name), 'wb') as save_file:
    for term in total_test_list:
        save_file.write((term + '\n').encode())
######################3DPW####################
base_path = "G:\\dataset\\3dpw\\"
image_path = "G:\\dataset\\3dpw\\imageFiles"
ttvs = os.listdir(base_path+'sequenceFiles')
ttvs.sort()
train_seq = []
test_seq = []
for section in ttvs:
    seq_path = os.path.join(base_path, 'sequenceFiles', section)
    sect = os.listdir(seq_path)
    sect.sort()
    for sequence in sect:
        if section[:4] == 'test':
            test_seq.append(sequence.split('.')[0])
        elif section[:4] == 'trai':
            train_seq.append(sequence.split('.')[0])
        else:
            train_seq.append(sequence.split('.')[0])

train_list = []
test_list = []

for seq in train_seq:
    list_path = os.path.join(image_path, seq)
    files_list = os.listdir(list_path)
    files_list.sort()
    for file in files_list[:length]:
        train_list.append(os.path.join('imageFiles', seq, file))
for seq in test_seq:
    list_path = os.path.join(image_path, seq)
    files_list = os.listdir(list_path)
    files_list.sort()
    for file in files_list[:length]:
        test_list.append(os.path.join('imageFiles', seq, file))

with open(os.path.join(base_path, train_name), 'wb') as save_file:
    for term in train_list:
        save_file.write((term + '\n').encode())
with open(os.path.join(base_path, test_name), 'wb') as save_file:
    for term in test_list:
        save_file.write((term + '\n').encode())
######################KITTI ODOMETRY####################
base_path = "G:\\dataset\\kitti\\"
image_path = "G:\\dataset\\kitti\\sequences"
sequences = os.listdir(image_path)
sequences.sort()

train_list = []
test_list = []

for seq in sequences[:8]:
    seq_path = os.path.join(image_path, seq, 'image_2')
    images = os.listdir(seq_path)
    images.sort()
    for image in images[:length]:
        train_list.append(os.path.join('sequences', seq, 'image_2', image))
for seq in sequences[8:11]:
    seq_path = os.path.join(image_path, seq, 'image_2')
    images = os.listdir(seq_path)
    images.sort()
    for image in images[:length]:
        test_list.append(os.path.join('sequences', seq, 'image_2', image))


with open(os.path.join(base_path, train_name), 'wb') as save_file:
    for term in train_list:
        save_file.write((term + '\n').encode())
with open(os.path.join(base_path, test_name), 'wb') as save_file:
    for term in test_list:
        save_file.write((term + '\n').encode())
print('s')
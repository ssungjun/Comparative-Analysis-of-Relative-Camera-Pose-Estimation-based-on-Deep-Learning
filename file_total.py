# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os
import shutil
######################7SCENES####################
base_path = "G:\\dataset\\7scenes\\"
total_list = []
situations = os.listdir(base_path)
situations.sort()
for situation in situations:
    seq_path = os.path.join(base_path, situation)
    seqs = os.listdir(seq_path)
    seqs.sort()
    seq_list = []
    for seq in seqs:
        if not seq[:3] == 'seq':
            continue
        files_path = os.path.join(seq_path, seq, 'image')
        seqs_list = os.listdir(files_path)
        seqs_list.sort()
        for file in seqs_list:
            seq_list.append(os.path.join(situation, seq, 'image', file))
    total_list = total_list + seq_list
    with open(os.path.join(seq_path, 'Total_file_list.txt'), 'wb') as save_file:
        for term in seq_list:
            save_file.write((term + '\n').encode())
with open(os.path.join(base_path, 'Total_file_list.txt'), 'wb') as save_file:
    for term in total_list:
        save_file.write((term + '\n').encode())

'''######################3DPW####################
base_path = "G:\\dataset\\3dpw\\"
image_path = "G:\\dataset\\3dpw\\imageFiles"
seqs = os.listdir(image_path)
seqs.sort()
total_seq = []
for seq in seqs:
    seq_path = os.path.join(image_path, seq)
    files = os.listdir(seq_path)
    files.sort()
    for file in files:
        total_seq.append(os.path.join('imageFiles', seq, file))

with open(os.path.join(base_path, 'Total_file_list.txt'), 'wb') as save_file:
    for term in total_seq:
        save_file.write((term + '\n').encode())'''
######################KITTI ODOMETRY####################
base_path = "G:\\dataset\\kitti\\"
image_path = "G:\\dataset\\kitti\\sequences"
sequences = os.listdir(image_path)
sequences.sort()

total_list = []

for seq in sequences[:11]:
    seq_path = os.path.join(image_path, seq, 'image_2')
    images = os.listdir(seq_path)
    images.sort()
    for image in images:
        total_list.append(os.path.join('sequences', seq, 'image_2', image))

with open(os.path.join(base_path, 'Total_file_list.txt'), 'wb') as save_file:
    for term in total_list:
        save_file.write((term + '\n').encode())

print('s')
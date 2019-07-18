from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement
import pickle
from pointnet.info3d import *
import random


#load pickled point cloud collection
def load(filename):
    with open(filename,'rb') as f: 
        point_collection = pickle.load(f)
       
    print(len(point_collection))

    """points_only_collection = []
    for label, pointCloud, triangles in centered_point_collection:
        remove normals
        points_only_collection.append(pointCloud[:,:3])
        print(label)
        print(pointCloud.shape)
        print(triangles.shape)

    print(len(points_only_collection))"""
    return point_collection


class HololensDataset(data.Dataset):
    def __init__(self, 
    split, 
    npoints=1000, 
    root='hololens_ransac_dataset.pickle',
    data_augmentation=True):
        
        self.npoints = npoints
        self.root = root
        self.split =split
        self.data_augmentation = data_augmentation
        

        self.pointcloudCollection = load(self.root)
        print("length ", len(self.pointcloudCollection))

        #create a list of classes using the class indexes
        self.classes = []
        for m in range(len(self.pointcloudCollection)):
            self.classes.append(m)

        print("sublength ", len(self.pointcloudCollection[0]))
        #generate triplets
        self.triplet_set, self.target_set = [], []
        #for each class (location)
        for k in range(len(self.pointcloudCollection)):
            #for each of the 20 ransac generalized clouds in the class
            for i in range(len(self.pointcloudCollection[k])):
                #for each of the 20 ransac generalized clouds in the class, again
                for j in range(int(len(self.pointcloudCollection[k]))):
                    #ensure that the same cloud doesnt go in as the first two clouds of the triplet
                    if i == j:
                        continue
                    triplet = []
                    #add the two clouds in the same class to the triplet
                    triplet.append(self.pointcloudCollection[k][i])
                    triplet.append(self.pointcloudCollection[k][j])

                    #pick a random class other than the original class
                    x = random.randint(0,len(self.pointcloudCollection)-1)
                    while x == k:
                        x = random.randint(0,len(self.pointcloudCollection)-1)
                    #pick a random cloud from the selected random class
                    y = random.randint(0, len(self.pointcloudCollection[x])-1)
                    #add the cloud from the different class to the triplet
                    triplet.append(self.pointcloudCollection[x][y])
                    self.triplet_set.append(triplet)
                    self.target_set.append([1,0])

        self.target_set = torch.tensor(self.target_set)
        #shuffle the entire dataset, since target_set has the same values, no need to shuffle it
        random.shuffle(self.triplet_set)

        #split into training and testing datasets
        print("triplet set length ", len(self.triplet_set))
        print("split ", len(self.triplet_set)*9/10)
        if self.split == 'train':
            self.triplet_set = self.triplet_set[:int(len(self.triplet_set)*9/10)]
        else:
            self.triplet_set = self.triplet_set[int(len(self.triplet_set)*9/10):]
                    
        print ("length of set ", len(self.triplet_set))

    #return one processed point cloud from triplet, cloud should be in 0->2
    def get_single_cloud(self, index, cloud):
        fn = self.triplet_set[index][cloud]
        #cls = self.classes[self.datapath[index][0]]
        point_set = np.array(fn, dtype=np.float32)
        #print(point_set.shape)
        #seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)

        choice = np.random.choice(len(point_set), self.npoints, replace=True)
        #minimum size of our i/p is 1000
        #resample (instead of taking all the points take only a selected number of points, default 1000 points)
        point_set = point_set[choice, :]

        #center and scale the point cloud
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        #seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        #seg = torch.from_numpy(seg)
        #cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        return point_set


    def __getitem__(self, index):
        point_sets = []
        for i in range (3):
            point_sets.append(self.get_single_cloud(index, i))
        
        point_sets = torch.stack(point_sets)
        target = self.target_set[index]
        return point_sets, target

    def __len__(self):
        return len(self.triplet_set)

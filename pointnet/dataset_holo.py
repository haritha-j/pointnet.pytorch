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


#load pickled point cloud collection
def load(filename):
    with open(filename,'rb') as f: 
        centered_point_collection = pickle.load(f)
       
    print(len(centered_point_collection))

    points_only_collection = []
    for label, pointCloud, triangles in centered_point_collection:
        #remove normals
        #points_only_collection.append(pointCloud[:,:3])
        print(label)
        print(pointCloud.shape)
        print(triangles.shape)

    #print(len(points_only_collection))
    return centered_point_collection


class HololensDataset(data.Dataset):
    def __init__(self, split, npoints=1000, root='point_collection/all_point_collection.pickle', ransac_iterations=50):
        self.npoints = npoints
        self.root = root
        self.split =split

        self.pointcloudCollection = load(self.root)

        #for each point cloud, create a number of examples by applying ransac.
        ransacCollection = []
        for location in self.pointcloudCollection:
            ransacCollectionForLocation = []
            for i in range ransac_iterations:
                ransacCloud, _ = getRansacPlanes(location, )


    #return one processed point cloud from triplet, cloud should be in 0->2
    def get_single_cloud(self, index, cloud):
        fn = self.triplet_set[index][cloud]
        #cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
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

        if self.classification:
            return point_set#, cls
        else:
            return point_set#, seg


    def __getitem__(self, index):
        point_sets = []
        for i in range (3):
            point_sets.append(self.get_single_cloud(index, i))
        
        point_sets = torch.stack(point_sets)
        target = self.target_set[index]
        return point_sets, target

    def __len__(self):
        return len(self.pointcloudCollection)

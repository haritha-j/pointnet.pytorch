from __future__ import print_function
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib as mpl
import math
import os
#from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
from scipy.spatial import Delaunay
#import cv2
import csv
import time
from generalizations import *
from sklearn.neighbors import NearestNeighbors
from info3d import *
import argparse
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import pointNetSiamese, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
from dataset_holo import *


def main():

    check_equal = False
    partial_realease = False
    rotate = False
    partial_release_radius = 4
    rotate_theta = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_points', type=int, default=2000, help='input batch size')
    parser.add_argument('--model', type=str, default='model/cls_model_new_ransac_249.pth', help='model path')
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

    opt = parser.parse_args()
    print(opt)
    num_points = opt.num_points

    pointcloud_collection = load('../utils/hololens_ransac_dataset/ransac_dataset_drop_0.2_30.pickle')

    #rotated_pointcloud_collection = rotatePointCollection(pointcloud_collection, rotate_theta)
    #load model
    classifier = pointNetSiamese(k=10, feature_transform=opt.feature_transform)#k is the number of classes in the training dataset
    classifier.load_state_dict(torch.load(opt.model, map_location=device))
    classifier.cuda()
    classifier.eval()

    correct = 0
    total = 0
    for i in range (len(pointcloud_collection)):
        #define original point cloud for comparison
        points_original = pointcloud_collection[i][0]
        results = []
        #compare against all other pointclouds in collection
        for j in range (len(pointcloud_collection)):

            if partial_realease:
                #get partial pointcloud
                points_new, _, _ = getPartialPointCloud(pointcloud_collection[j][1], pointcloud_collection[j][2], partial_release_radius)
                print ("new points length ", len(points_new))
                points_new = points_new[:,:3]
                num_points = len(points_new)
            elif rotate:
                #get rotated point cloud
                points_new = rotated_pointcloud_collection[j][0]
            else:
                points_new = pointcloud_collection[j][0]
            
            result = getInferenceScore(points_original, points_new, num_points, device, classifier)
            results.append(result)
        selection = results.index(max(results))
        print("count positive ", results.count(1))
        print("count negative ", results.count(0))
        print("selection ", selection, " correct class ", i)
        if selection == i:
            correct +=1
        total += 1
    print("correct ", correct)
    print("total ", total)
    print ("final accuracy = ", float(correct)/float(total))


def getInferenceScore (points_original, points_new, num_points, device, classifier):
    #one-shot inference
    with torch.no_grad():
        #points_new = points_new.to(device)
        #points_original = points_original.to(device)
        #resize points clouds to a standard size (1000 default)
        points_original = preparePointCloud(points_original, num_points, True)
        points_new = preparePointCloud(points_new, num_points, True)
        #choice_original = np.random.choice(len(points_original), num_points, replace=True)
        #choice_new = np.random.choice(len(points_new), num_points, replace=True)
        #points_original = points_original[choice_original, :]
        #points_new = points_new[choice_new, :]
        data = []
        image_pair = []
        #print ("points original", points_original.shape)
        image_pair.append(points_original)
        image_pair.append(points_new)
        image_pair = torch.stack(image_pair)
        data.append(image_pair) #create a dataset of only one image pair
        #data = np.asarray(data)
        #print("shape ", data.shape)
        #data = torch.from_numpy(data)
        
        data = torch.stack(data)
        data = data.transpose(2,3)
        #print("shape ", data.shape)
        data = data.float()
        data = data.to(device)
        output, _, _ = classifier(data)
        print ("output ", output)
        res1 = torch.argmax(output, dim=1)
        result = (output[0][1] - output[0][0]).cpu().item()
        #result = torch.squeeze(torch.argmax(output, dim=1)).cpu().item()
        

        print (res1)
        return (res1)


def preparePointCloud(pointcloud, num_points, data_augmentation):
    #cls = self.classes[self.datapath[index][0]]
    point_set = np.array(pointcloud, dtype=np.float32)

    choice = np.random.choice(len(point_set), num_points, replace=True)
    #minimum size of our i/p is 1000
    #resample (instead of taking all the points take only a selected number of points, default 1000 points)
    point_set = point_set[choice, :]

    #center and scale the point cloud
    point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
    point_set = point_set / dist #scale

    if data_augmentation:
        theta = np.random.uniform(0,np.pi*2)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
        point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

    #seg = seg[choice]
    point_set = torch.from_numpy(point_set)
    #seg = torch.from_numpy(seg)
    #cls = torch.from_numpy(np.array([cls]).astype(np.int64))

    return point_set


if __name__ == '__main__':
    main()  
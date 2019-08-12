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
from pointnet.model import pointNetSiamese, feature_transform_regularizer, pointNetParallel
import torch.nn.functional as F
from tqdm import tqdm
from dataset_holo import *


def main():

    check_equal = False
    partial_release = False
    rotate = True
    partial_release_radius = 3
    rotate_theta = 1
    compare_against_original = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_points', type=int, default=2000, help='input batch size')
    parser.add_argument('--model', type=str, default='model/cls_model_new_ransac_249.pth', help='model path')
    parser.add_argument('--model_type', type=str, default='siamese', help='siamese or parallel')
    parser.add_argument('--dataset', type=str, default='../utils/hololens_ransac_dataset/ransac_dataset_drop_0.2_30_with_triangles.pickle', help='dataset location')

    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

    opt = parser.parse_args()
    print(opt)
    num_points = opt.num_points

    pointcloud_collection = load(opt.dataset)
    num_classes = 73
    #rotated_pointcloud_collection = rotatePointCollection(pointcloud_collection, rotate_theta)
    #load model

    if opt.model_type == 'siamese':
        classifier = pointNetSiamese(k=num_classes, feature_transform=opt.feature_transform)
    else:
        classifier = pointNetParallel(k=num_classes, feature_transform=opt.feature_transform) 
    classifier.load_state_dict(torch.load(opt.model, map_location=device))
    if torch.cuda.is_available():
        classifier.cuda()
    classifier.eval()

    correct = 0
    total = 0

    new_point_clouds = []
    #compare against all other pointclouds in collection, currently using the originals, not ransac
    for j in range (len(pointcloud_collection)):
        if partial_release:
            #get partial pointcloud
            points_new, _, _ = getPartialPointCloud(pointcloud_collection[j][0][0], pointcloud_collection[j][0][1], partial_release_radius)
            print ("new points length ", len(points_new))
            points_new = points_new[:,:3]
            num_points = len(points_new)
        else:
            points_new = (pointcloud_collection[j][0][0][:,:3])
        new_point_clouds.append(points_new)

    for i in range (len(pointcloud_collection)):
        #define original point cloud for comparison
        if compare_against_original:
            points_original = pointcloud_collection[i][0][0][:,:3]
        else:
            
            points_original = random.choice(pointcloud_collection[i][1:])[0][:,:3]

            
        result = getInferenceScore(points_original, new_point_clouds, num_points, device, classifier, rotate)

        if result == i:
            correct +=1
        total += 1
    print("correct ", correct)
    print("total ", total)
    print ("final accuracy = ", float(correct)/float(total))


def getInferenceScore (points_original, new_point_clouds, num_points, device, classifier, rotate):
    #one-shot inference
    with torch.no_grad():
        #points_new = points_new.to(device)
        #points_original = points_original.to(device)
        #resize points clouds to a standard size (1000 default)
        points_original = preparePointCloud(points_original, num_points, True)
        data = []
        for i in range (len(new_point_clouds)):
            points_new = preparePointCloud(new_point_clouds[i], num_points, True)
            
            #extra check for rotational invariance
            randomTheta = (2*np.pi)/(random.randint(1,360))
            randomAxis = random.randint(0,2)
            if rotate:
                points_new = torch.from_numpy(rotatePointCloudWithoutNormals(points_new, randomTheta, randomAxis))
            image_pair = []
            image_pair.append(points_original)
            image_pair.append(points_new)
            image_pair = torch.stack(image_pair)
            data.append(image_pair)
        #choice_original = np.random.choice(len(points_original), num_points, replace=True)
        #choice_new = np.random.choice(len(points_new), num_points, replace=True)
        #points_original = points_original[choice_original, :]
        #points_new = points_new[choice_new, :]
        #data = np.asarray(data)
        #print("shape ", data.shape)
        #data = torch.from_numpy(data)
        data = torch.stack(data)
        data = data.transpose(2,3)
        #print("shape ", data.shape)
        data = data.float()
        data = data.to(device)
        output, _, _ = classifier(data)
        #print ("output ", output)
        result = torch.max(output, dim=1)

        #of all the positive results, pick the highest positive result
        max_index = 0
        max_value = 0
        #print("res", result)
        for j in range (len(result.indices)):
            if result.indices[j]==1:
                if result.values[j] > max_value:
                    max_index = j
                    max_value = result.values[j]
        
        if max_value == 0:
            print("no result found")
            return (-1)
        else:
            print("final result ", max_value, max_index)
            return (max_index)

        #result = (output[0][1] - output[0][0]).cpu().item()
        #result = torch.squeeze(torch.argmax(output, dim=1)).cpu().item()



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
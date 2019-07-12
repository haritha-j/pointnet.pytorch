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
    rotate = True
    partial_release_radius = 4
    rotate_theta = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_points', type=int, default=1000, help='input batch size')
    parser.add_argument('--model', type=str, default='model/siamese_ransac_holo_rotation_cls_model_19.pth', help='model path')
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

    opt = parser.parse_args()
    print(opt)
    num_points = opt.num_points

    pointcloud_collection = load('point_collection/all_point_collection.pickle')
    rotated_pointcloud_collection = rotatePointCollection(pointcloud_collection, rotate_theta)
    #load model
    classifier = pointNetSiamese(k=10, feature_transform=opt.feature_transform)#k is the number of classes in the training dataset
    classifier.load_state_dict(torch.load(opt.model, map_location=device))
    classifier.cuda()
    classifier.eval()

    correct = 0
    total = 0
    for i in range (len(pointcloud_collection)):
        #define original point cloud for comparison
        points_original = pointcloud_collection[i][1][:,:3]
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
                points_new = rotated_pointcloud_collection[j][1][:,:3]
            else:
                points_new = pointcloud_collection[j][1][:,:3]
            
            result = getInferenceScore(points_original, points_new, num_points, device, classifier)
            results.append(result)
        selection = results.index(max(results))
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
        choice_original = np.random.choice(len(points_original), num_points, replace=True)
        choice_new = np.random.choice(len(points_new), num_points, replace=True)
        points_original = points_original[choice_original, :]
        points_new = points_new[choice_new, :]
        data = []
        image_pair = []
        #print ("points original", points_original.shape)
        image_pair.append(points_original)
        image_pair.append(points_new)
        data.append(image_pair) #create a dataset of only one image pair
        data = np.asarray(data)
        #print("shape ", data.shape)
        data = torch.from_numpy(data)
        data = data.transpose(2,3)
        #print("shape ", data.shape)
        data = data.float()
        data = data.to(device)
        output, _, _ = classifier(data)
        print ("output ", output)
        result = (output[0][1] - output[0][0]).cpu().item()
        #result = torch.squeeze(torch.argmax(output, dim=1)).cpu().item()
        

        print (result)
        return (result)

if __name__ == '__main__':
    main()  
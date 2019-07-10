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
    threshold = 30
    check_equal = False
    partial_realease = False
    rotate = False
    partial_release_radius = 4
    rotate_theta = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_points', type=int, default=1000, help='input batch size')
    parser.add_argument('--model', type=str, default='model/siamese_corrected_249.pth', help='model path')
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

    opt = parser.parse_args()
    print(opt)
    num_points = opt.num_points

    #load data
    pointcloud_collection = load('point_collection/all_point_collection.pickle')
    rotated_pointcloud_collection = rotatePointCollection(pointcloud_collection, rotate_theta)
    #load model
    classifier = pointNetSiamese(k=10, feature_transform=opt.feature_transform)#k is the number of classes in the training dataset
    
    classifier.load_state_dict(torch.load(opt.model, map_location=device))
    #classifier.cuda()
    classifier.eval()

    correct = 0
    total = 0
    for i in range (len(pointcloud_collection)):
        #define original and new point cloud for comparison
        points_original = pointcloud_collection[i][1][:,:3]
        #print ("points original shape ", len(points_original[0]))
        #check the accuracy for the same splice (location)
        if check_equal:
            if partial_realease:
                #get partial pointcloud
                points_new, _, _ = getPartialPointCloud(pointcloud_collection[i][1], pointcloud_collection[i][2], partial_release_radius)
                print ("new points length ", len(points_new))
                points_new = points_new[:,:3]
                num_points = len(points_new)
            elif rotate:
                #get rotated point cloud
                points_new = rotated_pointcloud_collection[i][1][:,:3]
            else:
                points_new = pointcloud_collection[i][1][:,:3]
            
            result = infer(points_original, points_new, num_points, device, classifier, threshold)
            if result == 0:
                correct += 1
            total += 1
        #check accuracy for different splice (location)
        else:
            for j in range(len(pointcloud_collection)):
                if i==j:
                    continue
                else:
                    points_new = pointcloud_collection[j][1][:,:3]
                    result = infer(points_original, points_new, num_points, device, classifier, threshold)
                    if result == 1:
                        correct += 1
                    total += 1
    print ("correct ", correct, " incorrect ", total - correct )
    print (float(correct)/float(total))
                
def infer (points_original, points_new, num_points, device, classifier, threshold):
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
        print (output)
        result = torch.squeeze(torch.argmax(output, dim=1)).cpu().item()
        
        """
        #custom score
        score = (abs(output[0][0]) + abs(output[0][1])).cpu().item()
        if score > threshold:
            result = 1
        else:
            result = 0
        """
        print (result)
        return (result)
        
        



if __name__ == '__main__':
    main()  
#generate partial pointclouds from the original dataset (no ransac)

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
from pointnet.dataset_holo import load
from multiprocessing import Process, Manager

partial_release_radius = 2
no_of_partial_clouds = 100

def get_partial_clouds_for_location(location, triangles, no_of_partial_clouds, partial_release_radius, partial_cloud_collection):
    partial_clouds = []
    #first item is the actual point cloud, second item onwards are the ransac planes
    partial_clouds.append([location, triangles])
    for i in range(no_of_partial_clouds):
        points_new, triangles_new, _ = getPartialPointCloud(location, triangles, partial_release_radius)
        print ("generated cloud no. ", i, " of size ", len(points_new))
        partial_clouds.append([points_new, triangles_new])
    partial_cloud_collection.append(partial_clouds)
    print ("current length ", len(partial_cloud_collection))

def main():
    root='../pointnet/point_collection/all_point_collection.pickle'
    pointcloudCollection = load(root)
    print("length", len(pointcloudCollection))

    locationsOnly = []
    trianglesOnly = []
    for _, location, triangle in pointcloudCollection:
        #print (len(location[0]))
        locationsOnly.append(location)
        trianglesOnly.append(triangle)

    with Manager() as manager:
        partialCloudCollection = manager.list()
        processes = []
        
        for k in range(len(locationsOnly)):
            print ("processing location ", k, " with original length ", len(locationsOnly[k]))
            p = Process(target=get_partial_clouds_for_location, args=(locationsOnly[k], trianglesOnly[k], no_of_partial_clouds, partial_release_radius, partialCloudCollection))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        partialCloudCollection = list(partialCloudCollection)
        print(len(partialCloudCollection))
        print(len(partialCloudCollection[0]))
        print(len(partialCloudCollection[0][0]))

        with open("partial_dataset_radius_1_x10_with_triangles.pickle", 'wb') as f:
            pickle.dump(partialCloudCollection, f)


if __name__ == '__main__':
    main()
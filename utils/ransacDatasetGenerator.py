#generate ransac variations from original dataset
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
   
#get multiple point clouds for a given pointcloud by applying ransac
def getRansacPointCloudsforLocation(location, triangle, ransac_copies, ransacCollection, point_drop_threshold, trials):
    ransacCollectionForLocation = []
    for i in range (ransac_copies):
        ransacPlanes, ransacPlaneProperties = getRansacPlanes(location, trials=trials)
        ransacCloud, _ = getGeneralizedPointCloud(ransacPlanes, ransacPlaneProperties, point_drop_threshold)
        print("ransac cloud length ", len(ransacCloud))
        ransacCollectionForLocation.append(ransacCloud[:,:3])
        print(len(ransacCloud[:,:3][0]))
    ransacCollection.append(ransacCollectionForLocation)
    print ("current length ", len(ransacCollection))

#get a ransac collection with both point clouds and triangles for testing partial release
def getRansacPointCloudsforLocationWithTriangles(location,triangle, ransac_copies, ransacCollection, point_drop_threshold, trials):
    #first item is the actual point cloud, second item onwards are the ransac planes
    ransacCollectionForLocation = []
    ransacCollectionForLocation.append([location, triangle])
    for i in range (ransac_copies):
        ransacPlanes, ransacPlaneProperties = getRansacPlanes(location, trials=trials)
        ransacCloud, triangles = getGeneralizedPointCloud(ransacPlanes, ransacPlaneProperties, point_drop_threshold)
        print("ransac cloud length ", len(ransacCloud))
        ransacCollectionForLocation.append([ransacCloud, triangles])
        print(len(ransacCloud[:,:3][0]))
    ransacCollection.append(ransacCollectionForLocation)
    print ("current length ", len(ransacCollection))

#for each point cloud, create a number of examples by applying ransac.
def main():
    ransac_copies = 10
    point_drop_threshold = 0.2
    trials = 30
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
        ransacCollection = manager.list()
        processes = []
        i=0
        for k in range(len(locationsOnly)):
            """if i>3:
                break"""
            p = Process(target=getRansacPointCloudsforLocationWithTriangles, args=(locationsOnly[k], trianglesOnly[k], ransac_copies, ransacCollection, point_drop_threshold, trials))
            p.start()
            processes.append(p)
            i+=1

        for p in processes:
            p.join()

        ransacCollection = list(ransacCollection)
        print("threads complete")
        print (len(ransacCollection))
        print (len(ransacCollection[0]))
        print (len(ransacCollection[0][0]))

        with open('ransac_dataset_drop_0.2_30_with_triangles.pickle','wb') as f:
            pickle.dump(ransacCollection,f)


if __name__ == '__main__':
    main()
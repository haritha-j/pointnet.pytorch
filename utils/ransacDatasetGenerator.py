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
def getRansacPointCloudsforLocation(location, ransac_iterations, ransacCollection, point_drop_threshold):
    ransacCollectionForLocation = []
    for i in range (ransac_iterations):
        ransacPlanes, ransacPlaneProperties = getRansacPlanes(location)
        ransacCloud, _ = getGeneralizedPointCloud(ransacPlanes, ransacPlaneProperties, point_drop_threshold)
        print("ransac cloud length ", len(ransacCloud))
        ransacCollectionForLocation.append(ransacCloud[:,:3])
        print(len(ransacCloud[:,:3][0]))
    ransacCollection.append(ransacCollectionForLocation)
    print ("current length ", len(ransacCollection))


#for each point cloud, create a number of examples by applying ransac.
def main():
    ransac_iterations = 20
    point_drop_threshold = 0.2
    root='../pointnet/point_collection/all_point_collection.pickle'
    pointcloudCollection = load(root)
    print("length", len(pointcloudCollection))

    locationsOnly = []
    for _, location, _ in pointcloudCollection:
        #print (len(location[0]))
        locationsOnly.append(location)

    with Manager() as manager:
        ransacCollection = manager.list()
        processes = []
        i=0
        for location in locationsOnly:
            """if i>4:
                break"""
            p = Process(target=getRansacPointCloudsforLocation, args=(location, ransac_iterations, ransacCollection, point_drop_threshold))
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

        with open('ransac_dataset.pickle','wb') as f:
            pickle.dump(ransacCollection,f)


if __name__ == '__main__':
    main()
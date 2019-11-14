from __future__ import print_function
import argparse
import os
import random
import torch
from pointnet.info3d import *
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
#from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import pointNetSiamese, feature_transform_regularizer, pointNetParallel
from pointnet.dataset_holo import HololensDataset
from pointnet.dataset_holo_partial import HololensPartialDataset
import torch.nn.functional as F
from tqdm import tqdm

import os.path
import torch.utils.data as data
import numpy as np
import sys
import json
from plyfile import PlyData, PlyElement
import pickle
from pointnet.info3d import *
import random


class ShapeNetDatasetEval(data.Dataset):
    def __init__(self,
                 root,
                 npoints=1000,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True,
                 batch_size = 32,
                 holes=0,
                 hole_radius=0):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}
        self.batch_size = batch_size

        print ( "root")
        print (self.root)
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid+'.pts'),
                                        os.path.join(self.root, category, 'points_label', uuid+'.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print(self.seg_classes, self.num_seg_classes)

        print ("lenght of data ")
        print (len(self.datapath))
        #create point cloud sets of 16
        self.triplet_set, self.target_set = [], []
        #for each class
        for cloud_index in range(len(self.classes)):
            #make 4 sets of pointclouds
            for h in range(4):
                point_sets = []
                #added_indeces = []
                #in each set, there are 33 clouds of the same class, and first and second clouds are identical
                random_choice0 = np.random.choice(len(self.datapath))
                while ((self.classes[self.datapath[random_choice0][0]] != cloud_index)):
                    random_choice0 = np.random.choice(len(self.datapath))
                point_cloud = self.datapath[random_choice0]
                point_sets.append(point_cloud)
                point_sets.append(point_cloud)
                for j in range(batch_size-1):
                    random_choice1 = np.random.choice(len(self.datapath))
                    while ((self.classes[self.datapath[random_choice1][0]] != cloud_index) or (random_choice1 == random_choice0)):
                        random_choice1 = np.random.choice(len(self.datapath))
                        #print("S")
                    point_cloud = self.datapath[random_choice1]
                    #print("X")
                    point_sets.append(point_cloud)    


                self.triplet_set.append(point_sets) #should this be torch.stack'ed?
                #self.target_set.append([1,0])
        
        #self.target_set = torch.tensor(self.target_set)



    #return one processed point cloud from triplet, cloud should be in 0->32
    def get_single_cloud(self, index, cloud, augmentation):
        fn = self.triplet_set[index][cloud]
        #cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        #print("shapes", point_set.shape, seg.shape)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #minimum size of our i/p is 1000
        #resample (instead of taking all the points take only a selected number of points, default 1000 points)
        point_set = point_set[choice, :]

        #center and scale the point cloud
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if augmentation:
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
        for i in range (self.batch_size+1):
            if i ==0:
                point_sets.append(self.get_single_cloud(index, i, True))
            else:
                point_sets.append(self.get_single_cloud(index, i, False))
        
        point_sets = torch.stack(point_sets)
        #target = self.target_set[index]
        return point_sets#, target

    def __len__(self):
        return len(self.triplet_set)


def main():

    #rotational invariance built into data augmentation during data loading 
    rotate = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument(
        '--num_points', type=int, default=1000, help='input batch size')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--model', type=str, default='', help='model path')
    #parser.add_argument('--model_type', type=str, default='siamese', help='siamese or parallel')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
    parser.add_argument('--feature_transform', default=True, action='store_true', help="use feature transform")
    parser.add_argument('--holes', type=int, default=0)
    parser.add_argument('--hole_radius', type=int, default=0)


    opt = parser.parse_args()
    print(opt)

    blue = lambda x: '\033[94m' + x + '\033[0m'
   

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    batch_size = opt.batchSize

    if opt.dataset_type == 'shapenet':
        dataset = ShapeNetDatasetEval(
            root=opt.dataset,
            classification=True,
            npoints=opt.num_points,
            batch_size = batch_size,
            holes = opt.holes,
            hole_radius = opt.hole_radius)

        test_dataset = ShapeNetDatasetEval(
            root=opt.dataset,
            classification=True,
            split='test',
            npoints=opt.num_points,
            data_augmentation=True,
            batch_size = batch_size,
            holes = opt.holes,
            hole_radius = opt.hole_radius)
 
        
    else:
        exit('wrong dataset type')


    print("length ")
    print (len(test_dataset))
    num_classes = len(test_dataset.classes)
    print(num_classes)
    classifier = pointNetSiamese(k=num_classes, feature_transform=opt.feature_transform)
    classifier.load_state_dict(torch.load(opt.model, map_location=device))


    
    '''dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))
    
    testdataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers))

    print(len(dataset))
    num_classes = len(dataset.classes)
    print('classes', num_classes)

    try:
        #vvvvos.makedirs(opt.outf)
    except OSError:
        pass

    '''

    #if opt.model != '':
    #    classifier.load_state_dict(torch.load(opt.model))
        

    #optimizer = optim.Adam(classifier.parameters(), lr=0.0001, betas=(0.9, 0.999))
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    if torch.cuda.is_available():
        classifier.cuda()
    classifier = classifier.eval()

    #num_batch = len(dataset) / opt.batchSize
    correct_count = 0
    top3_count = 0
    total = 0
    for count in range(len(test_dataset)):
        print(count)
        #print ("train accurate labels, positive and negative ", accurate_labels_positive,accurate_labels_negative)
        #pick a random batch from the test dataset
        points = []
        target = []
        
        #create 32 pairs, where the first one contains a positive example and the rest are negative
        for j in range(1,batch_size+1):
            point_cloud_pair = []
            point_cloud_pair.append(test_dataset[count][0])
            point_cloud_pair.append(test_dataset[count][j])
            points.append(torch.stack(point_cloud_pair))
            if j == 1:
                target.append([1])
            else:
                target.append([0])
        points = torch.stack(points)
        target = torch.tensor(target)
        
        points = points.transpose(3, 2)
        #points_positive = points[:,:2]
        #points_negative = points[:,0:3:2]
        target = torch.squeeze(target)
        #target_negative = torch.squeeze(target[:,1])
        #print("SHape")
        #print(target)
        if torch.cuda.is_available():
            points = points.cuda()
            target = target.cuda()
            #points_positive, points_negative = points_positive.cuda(), points_negative.cuda()
            #target_negative, target_positive = target_negative.cuda(), target_positive.cuda()
        
        #print("shape of input", points_positive.shape)
        pred, trans, trans_feat = classifier(points) # original and positive image
        #pred_negative, trans, trans_feat = classifier(points_negative) # original and negative image
        #loss = F.cross_entropy(pred, target)        
        #loss_negative = F.cross_entropy(pred_negative, target_negative)
        #loss = loss_negative + loss_positive
        #print ("loss ", loss)
        #print("positive prediction ", pred_positive, " positive target ", target_positive, " positive loss ", loss_positive)
        #print("negative prediction ", pred_negative, " negative target ", target_negative, " negative loss ", loss_negative)

        #pred_choice = pred.data.max(1)[1]
        #correct = pred_choice.eq(target.data).cpu().sum()
        #print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))

        print ("pred")
        print (pred)
        
        
        result = torch.max(pred, dim=1)
        print("result")
        print(result.indices)
        #of all the positive results, pick the highest positive result and the top 3 positive results
        #we expect the top result to be the 0th index
        positives = []
        positive_indices = []
        max_index, max_index2, max_index3 = -1, -1, -1
        #print("res", result)
        for j in range (len(result.indices)):
            if result.indices[j]==1:
                positives.append(result.values[j].item())
                positive_indices.append(j)
        

        if len(positives)>0:
            print("max")
            print(positives)
            max_index = positive_indices[positives.index(max(positives))]

            del positive_indices[positives.index(max(positives))]
            del positives[positives.index(max(positives))]
            if (len(positives)>0):
                max_index2 = positive_indices[positives.index(max(positives))]
                del positive_indices[positives.index(max(positives))]
                del positives[positives.index(max(positives))]
                if (len(positives)>0):
                    max_index3 = positive_indices[positives.index(max(positives))]
            print ("top 3")
            print(max_index, max_index2, max_index3)
        else:
            print("no result found")

        if (max_index ==0):
            correct_count+=1
            print("result correct")
        if ((max_index == 0) or (max_index2 == 0) or (max_index3 == 0)):
            top3_count +=1
            print("result in top3")
        total+=1
        #accurate_test_labels_positive = torch.sum(torch.argmax(pred_positive, dim=1) == target_positive).cpu()
        #accurate_test_labels_negative = torch.sum(torch.argmax(pred_negative, dim=1) == target_negative).cpu()
        #accurate_test_labels = accurate_test_labels_positive + accurate_test_labels_negative
        #all_test_labels = len(target_positive) + len(target_negative)
        #test_accuracy = 100. * float(accurate_test_labels) / float(all_test_labels)
        #accuracy+= test_accuracy
        #print accuracy
        #print (accurate_test_labels_positive, accurate_test_labels_negative)
        #accuracy = 100. * float(accurate_labels) / float(all_labels)
        #print('Test accuracy: {}/{} ({:.3f}%)'.format(accurate_test_labels, all_test_labels, test_accuracy))
        #outputfile.write('Test accuracy: {}/{} ({:.3f}%)'.format(accurate_test_labels, all_test_labels, test_accuracy))
        #outputfile.write('Train accuracy: {}/{} ({:.3f}%)'.format(accurate_labels, all_labels, accuracy))
        #print('Train accuracy: {}/{} ({:.3f}%)'.format(accurate_labels, all_labels, accuracy))
        #outputfile.write("\n\n\n")

        #torch.save(classifier.state_dict(), '%s/cls_model_partial_ransac_lr_0001_partial_radius_1_parallel_%d.pth' % (opt.outf, epoch))
    print ("total")
    print(total)
    print("Correct")
    print(correct_count)
    print("top3")
    print(top3_count)
    print ("Accuracy")
    print(100. * float(correct_count)/float(total))
    print ("Top 3 Accuracy")
    print(100. * float(top3_count)/float(total))
    
    
    #print("final accuracy")
    #print(count)
    #print (accuracy/count)
"""
    total_correct = 0
    total_testset = 0
    for i,data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        #pred_choice = pred.data.max(1)[1]
        #correct = pred_choice.eq(target.data).cpu().sum()
        #total_correct += correct.item()
        #total_testset += points.size()[0]

    #print("final accuracy {}".format(total_correct / float(total_testset)))
"""
if __name__ == '__main__':
    main()  
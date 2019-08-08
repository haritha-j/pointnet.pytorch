from __future__ import print_function
import argparse
import os
import random
import torch
from pointnet.info3d import *
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import pointNetSiamese, feature_transform_regularizer, pointNetParallel
from pointnet.dataset_holo import HololensDataset
from pointnet.dataset_holo_partial import HololensPartialDataset
import torch.nn.functional as F
from tqdm import tqdm


def main():

    #rotational invariance built into data augmentation during data loading 
    rotate = False

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument(
        '--num_points', type=int, default=1000, help='input batch size')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument(
        '--nepoch', type=int, default=250, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--model_type', type=str, default='siamese', help='siamese or parallel')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

    opt = parser.parse_args()
    print(opt)

    blue = lambda x: '\033[94m' + x + '\033[0m'

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset_type == 'shapenet':
        dataset = ShapeNetDataset(
            root=opt.dataset,
            classification=True,
            npoints=opt.num_points)

        test_dataset = ShapeNetDataset(
            root=opt.dataset,
            classification=True,
            split='test',
            npoints=opt.num_points,
            data_augmentation=True)
    elif opt.dataset_type == 'modelnet40':
        dataset = ModelNetDataset(
            root=opt.dataset,
            npoints=opt.num_points,
            split='trainval')

        test_dataset = ModelNetDataset(
            root=opt.dataset,
            split='test',
            npoints=opt.num_points,
            data_augmentation=False)
    elif opt.dataset_type == 'hololens':
        dataset = HololensDataset(
            root=opt.dataset,
            npoints=opt.num_points,
            split='train')
        
        test_dataset = HololensDataset(
            root=opt.dataset,
            npoints=opt.num_points,
            split='test'
        )
    elif opt.dataset_type == 'hololensPartial':
        dataset = HololensPartialDataset(
            root=opt.dataset,
            npoints=opt.num_points,
            split='train')
        
        test_dataset = HololensPartialDataset(
            root=opt.dataset,
            npoints=opt.num_points,
            split='test'
        )
    
        
    else:
        exit('wrong dataset type')

    batch_size = opt.batchSize
    dataloader = torch.utils.data.DataLoader(
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
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.model_type == 'siamese':
        classifier = pointNetSiamese(k=num_classes, feature_transform=opt.feature_transform)
    else:
        classifier = pointNetSiamese(k=num_classes, feature_transform=opt.feature_transform) 

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))


    optimizer = optim.Adam(classifier.parameters(), lr=0.0001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    if torch.cuda.is_available():
        classifier.cuda()

    num_batch = len(dataset) / opt.batchSize

    for epoch in range(opt.nepoch):
        outputfile = open("output.txt", "w")
        scheduler.step()
        accurate_labels = 0
        all_labels = 0
        for i, data in enumerate(dataloader, 0):    
            #check the contents of data
            points, target = data
            #print ("sample data", points[0][0], len(points[0]))
            #print ("target")
            #print (target.shape)
            #print ("points")
            #print (points.shape)
            #if rotation is enabled, rotate positive and negative examples before starting training
            if rotate:
                for cloud in points:
                    randomTheta = (2*np.pi)/(random.randint(1,360))
                    randomAxis = random.randint(0,2)
                    if opt.dataset_type == 'hololens':
                        cloud[1] = torch.from_numpy(rotatePointCloudWithoutNormals(cloud[1], randomTheta, randomAxis))
                        cloud[2] = torch.from_numpy(rotatePointCloudWithoutNormals(cloud[1], randomTheta, randomAxis))
                    else:
                        cloud[1] = rotatePointCloud(cloud[1], randomTheta, randomAxis)
                        cloud[2] = rotatePointCloud(cloud[1], randomTheta, randomAxis)

            points = points.transpose(3, 2) #transpose xyz dimension with point dimension
            #print("points transposed")
            #print (points.shape)
            points_positive = points[:,:2]
            points_negative = points[:,0:3:2]
            target_positive = torch.squeeze(target[:,0])
            target_negative = torch.squeeze(target[:,1])
            if torch.cuda.is_available():
                points_positive, points_negative = points_positive.cuda(), points_negative.cuda()
                target_negative, target_positive = target_negative.cuda(), target_positive.cuda()
            #print("input to network", points_positive.shape)

            optimizer.zero_grad()
            classifier = classifier.train()
            pred_positive, trans, trans_feat = classifier(points_positive) # original and positive image
            pred_negative, trans, trans_feat = classifier(points_negative) # original and negative image

            #print("positive prediction ", pred_positive, " negative prediction ", pred_negative)

            loss_positive = F.cross_entropy(pred_positive, target_positive)
            loss_negative = F.cross_entropy(pred_negative, target_negative)
            loss = loss_negative + loss_positive
            #loss = F.nll_loss(pred, target)
            
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()

            #check accuracy
            accurate_labels_positive = torch.sum(torch.argmax(pred_positive, dim=1) == target_positive).cpu()
            accurate_labels_negative = torch.sum(torch.argmax(pred_negative, dim=1) == target_negative).cpu()
            accurate_labels = accurate_labels + accurate_labels_positive + accurate_labels_negative
            all_labels = all_labels + len(target_positive) + len(target_negative)
            outputfile.write("epoch "+ str(epoch) + " i " +  str(i)+ "\n")
            outputfile.write("accurate_labels_positive "+ str(accurate_labels_positive))
            outputfile.write("accurate_labels_negative "+ str(accurate_labels_negative))
            outputfile.write("all_labels "+ str(all_labels))
            outputfile.write("\n")
            
            #pred_choice = pred.data.max(1)[1]
            #correct = pred_choice.eq(target.data).cpu().sum()
            #print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

            if i % 100 == 0:
                print("epoch ", epoch, " i ", i)
                #print ("train accurate labels, positive and negative ", accurate_labels_positive,accurate_labels_negative)
                #pick a random batch from the test dataset
                points = []
                target = []
                for j in range(batch_size):
                    k = random.randint(0,len(test_dataset)-1)
                    points.append(test_dataset[k][0])
                    target.append(test_dataset[k][1])
                points = torch.stack(points)
                target = torch.stack(target)
                
                points = points.transpose(3, 2)
                points_positive = points[:,:2]
                points_negative = points[:,0:3:2]
                target_positive = torch.squeeze(target[:,0])
                target_negagtive = torch.squeeze(target[:,1])
                if torch.cuda.is_available():
                    points_positive, points_negative = points_positive.cuda(), points_negative.cuda()
                    target_negagtive, target_positive = target_negagtive.cuda(), target_positive.cuda()
                
                #print("shape of input", points_positive.shape)
                classifier = classifier.eval()
                pred_positive, trans, trans_feat = classifier(points_positive) # original and positive image
                pred_negative, trans, trans_feat = classifier(points_negative) # original and negative image
                loss_positive = F.cross_entropy(pred_positive, target_positive)
                print("positive prediction ", pred_positive, " positive target ", target_positive, " positive loss ", loss_positive)
                print("negative prediction ", pred_negative, " negative target ", target_negative, " negative loss ", loss_negative)
        
                loss_negative = F.cross_entropy(pred_negative, target_negagtive)
                loss = loss_negative + loss_positive
                print ("loss ", loss)
                #pred_choice = pred.data.max(1)[1]
                #correct = pred_choice.eq(target.data).cpu().sum()
                #print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))

                accurate_test_labels_positive = torch.sum(torch.argmax(pred_positive, dim=1) == target_positive).cpu()
                accurate_test_labels_negative = torch.sum(torch.argmax(pred_negative, dim=1) == target_negative).cpu()
                accurate_test_labels = accurate_test_labels_positive + accurate_test_labels_negative
                all_test_labels = len(target_positive) + len(target_negative)
                test_accuracy = 100. * float(accurate_test_labels) / float(all_test_labels)

                #print accuracy
                print (accurate_test_labels_positive, accurate_test_labels_negative)
                accuracy = 100. * float(accurate_labels) / float(all_labels)
                print('Test accuracy: {}/{} ({:.3f}%)'.format(accurate_test_labels, all_test_labels, test_accuracy))
                outputfile.write('Test accuracy: {}/{} ({:.3f}%)'.format(accurate_test_labels, all_test_labels, test_accuracy))
                outputfile.write('Train accuracy: {}/{} ({:.3f}%)'.format(accurate_labels, all_labels, accuracy))
                print('Train accuracy: {}/{} ({:.3f}%)'.format(accurate_labels, all_labels, accuracy))
                outputfile.write("\n\n\n")

        torch.save(classifier.state_dict(), '%s/cls_model_new_ransac_lr_0001_partial_radius_1_%d.pth' % (opt.outf, epoch))
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
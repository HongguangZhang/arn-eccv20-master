import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import meta_task_generator_test as mtg
import os
import math
import argparse
import scipy as sp
import scipy.stats
import time
import src.models as models

parser = argparse.ArgumentParser(description="Action Relation Network")
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 5)
parser.add_argument("-b","--query_num_per_class",type = int, default = 3)
parser.add_argument("-e","--episode",type = int, default= 100000)
parser.add_argument("-t","--query_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
parser.add_argument("-d","--dataset",type=str,default='hmdb51')
parser.add_argument("-m","--method",type=str,default='sa_jigsaw')
parser.add_argument("-f","--frame",type=int,default=20)
parser.add_argument("-sigma","--sigma",type=float,default=100.0)
parser.add_argument("-alpha","--alpha",type=float,default=0)
parser.add_argument("-beta","--beta",type=float,default=0)
parser.add_argument("-jigsaw","--jigsaw",type=int,default=10)
args = parser.parse_args()


# Hyper Parameters
METHOD = "arn_" + args.method
CLASS_NUM = args.class_num
SUPPORT_NUM_PER_CLASS = args.sample_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.query_episode
LEARNING_RATE = args.learning_rate
HIDDEN_UNIT = args.hidden_unit
SIGMA = args.sigma
TEMPORAL_DIM = np.int(np.floor(args.frame/4.0))

Log_Name = 'logs/' + METHOD + '_' + str(args.sample_num_per_class) + 'shot.txt'
if os.path.exists('logs') == False:
	os.system('mkdir logs')
if os.path.exists(Log_Name) == False:
	create_file = open(Log_Name, "w")
	create_file.close()
f = open(Log_Name, "r+")
f.write(METHOD+"\n")
f.close()

def power_norm(x, SIGMA):
	out = 2.0*F.sigmoid(SIGMA*x) - 1.0
	return out
	
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_folders,metaquery_folders = mtg.generate_folders(args.dataset)

    # Step 2: init neural networks
    print("init neural networks")

    n = torch.cuda.device_count()
    feature_encoder = nn.DataParallel(models.FeatureEncoder3D().apply(weights_init).cuda(), device_ids=list(range(n)), output_device=0)
    relation_network = nn.DataParallel(models.SimilarityNetwork(64,8).apply(weights_init).cuda(), device_ids=list(range(n)), output_device=0)
    jigsaw_discriminator = nn.DataParallel(models.Discriminator3D(int(args.jigsaw)).apply(weights_init).cuda(), device_ids=list(range(n)), output_device=0)
    spatial_detector = nn.DataParallel(models.SpatialDetector().apply(weights_init).cuda(), device_ids=list(range(n)), output_device=0)

    optimizer = torch.optim.Adam([{'params': feature_encoder.parameters()}, 
                                 {'params': relation_network.parameters()},
                                 {'params': jigsaw_discriminator.parameters()},
                                 {'params': spatial_detector.parameters()}], lr=LEARNING_RATE)

    optimizer_scheduler = StepLR(optimizer,step_size=50000,gamma=0.5)
    
    if os.path.exists("checkpoints/" + METHOD + "/chechpoint_" + str(SUPPORT_NUM_PER_CLASS) + "shot.pth.tar"):
    	checkpoint = torch.load("checkpoints/" + METHOD + "/chechpoint_" + str(SUPPORT_NUM_PER_CLASS) + "shot.pth.tar")
    	feature_encoder.load_state_dict(checkpoint['feature_encoder'])
    	relation_network.load_state_dict(checkpoint['relation_network'])
    	jigsaw_discriminator.load_state_dict(checkpoint['jigsaw_discriminator'])
    	spatial_detector.load_state_dict(checkpoint['spatial_detector'])
    	print("load modules successfully!")

    if os.path.exists("checkpoints/" + METHOD) == False:
        os.system('mkdir ' + "checkpoints/" + METHOD)

    best_accuracy = 0.0
    start = time.time()

    for episode in range(20):
        feature_encoder.eval()
        relation_network.eval()
        jigsaw_discriminator.eval()
        spatial_detector.eval()
        with torch.no_grad():         
            # query
            print("Testing...")
            accuracies = []
            for i in range(TEST_EPISODE):
                total_rewards = 0
                counter = 0
                task = mtg.ARTask(metaquery_folders,CLASS_NUM,SUPPORT_NUM_PER_CLASS,15,args.frame)
                sample_dataloader = mtg.get_ar_data_loader(task,num_per_class=SUPPORT_NUM_PER_CLASS,split="train",shuffle=False)
                num_per_class = 5
                query_dataloader = mtg.get_ar_data_loader(task,num_per_class=num_per_class,split="test",shuffle=False)

                sample_images,sample_labels,sample_frames = sample_dataloader.__iter__().next()
                for query_images,query_labels,query_frames in query_dataloader:
                    with torch.no_grad():
                        query_size = query_labels.shape[0]
                        sample_images = Variable(sample_images).permute(0,2,1,3,4).cuda()
                        query_images = Variable(query_images).permute(0,2,1,3,4).cuda()
                        
                        sample_features = feature_encoder(sample_images)
                        query_features = feature_encoder(query_images)
                        
                        sample_ta = 1 + spatial_detector(sample_features)
                        query_ta = 1 + spatial_detector(query_features)
                        
                        sample_features = (sample_features*sample_ta).view(SUPPORT_NUM_PER_CLASS*CLASS_NUM,64,TEMPORAL_DIM*32*32)
                        query_features = (query_features*query_ta).view(num_per_class*CLASS_NUM,64,TEMPORAL_DIM*32*32)
                        
                        so_sample_features = Variable(torch.Tensor(CLASS_NUM*SUPPORT_NUM_PER_CLASS, 1, 64, 64)).cuda()
                        so_query_features = Variable(torch.Tensor(num_per_class*CLASS_NUM, 1, 64, 64)).cuda()
                        
                        for dd in range(sample_images.size(0)):
                            s = sample_features[dd,:,:].view(64,-1)
                            s = (1.0 / s.size(1)) * s.mm(s.t())
                            so_sample_features[dd,:,:,:] = power_norm(s / s.trace(), SIGMA)
                        for dd in range(query_images.size(0)):
                            t = query_features[dd,:,:].view(64,-1)
                            t = (1.0 / t.size(1)) * t.mm(t.t())
                            so_query_features[dd,:,:,:] = power_norm(t / t.trace(), SIGMA)
                            
                        so_sample_features = so_sample_features.view(CLASS_NUM,SUPPORT_NUM_PER_CLASS,1,64,64).mean(1) # generate second-order support prototypes

                        sample_features_ext = so_sample_features.unsqueeze(0).repeat(query_size,1,1,1,1)

                        query_features_ext = so_query_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
                        query_features_ext = torch.transpose(query_features_ext,0,1)
                        relation_pairs = torch.cat((sample_features_ext,query_features_ext),2).view(-1,2,64,64)
                        relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

                        _,predict_labels = torch.max(relations.data,1)

                        rewards = [1 if predict_labels[j]==query_labels[j].cuda() else 0 for j in range(query_size)]

                        total_rewards += np.sum(rewards)
                        counter += query_size

                accuracy = total_rewards/1.0/counter
                accuracies.append(accuracy)

            test_accuracy,h = mean_confidence_interval(accuracies)

            f = open(Log_Name, "a")
            f.write("episode: %d----test acc:%.4f  h:%.4f----best acc:%.4f\n"%(episode+1,test_accuracy,h, best_accuracy))
            f.close()
            
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                
            print("episode:",episode+1, "    test accuracy:",test_accuracy," h:",h, "    best acc:", best_accuracy)


if __name__ == '__main__':
    main()

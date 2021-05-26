import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader,Dataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler

def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x

def generate_folders(dataset):
    train_folder = './datas/' + dataset + '/train'
    test_folder = './datas/' + dataset + '/val'

    metatrain_folders = [os.path.join(train_folder, label) \
                for label in os.listdir(train_folder) \
                if os.path.isdir(os.path.join(train_folder, label)) \
                ]
    metatest_folders = [os.path.join(test_folder, label) \
                for label in os.listdir(test_folder) \
                if os.path.isdir(os.path.join(test_folder, label)) \
                ]

    random.seed(1)
    random.shuffle(metatrain_folders)
    random.shuffle(metatest_folders)

    return metatrain_folders,metatest_folders

class ARTask(object):

    def __init__(self, character_folders, num_classes, train_num, test_num, frame_num):

        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num
        self.frame_num = frame_num

        class_folders = random.sample(self.character_folders,self.num_classes)
        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
        samples = dict()

        self.train_roots = []
        self.test_roots = []
        for c in class_folders:

            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num+test_num]

        self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]
        self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]
        

    def get_class(self, sample):
        return os.path.join(*sample.split('/')[:-1])


class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class ARDataset(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(ARDataset, self).__init__(*args, **kwargs)
        self.frame_num = int(args[0].frame_num)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        img_name = (os.listdir(image_root))
        img_name.sort()
        frame = self.frame_num
        video = torch.Tensor(frame, 3, 128, 128)
        if len(img_name) > frame:
        	img_name = random.sample(img_name, frame)
        	img_name.sort()
        	num_frame = frame
        else:
        	num_frame = len(img_name)
        	
        for n in range(len(img_name)):
        	image = Image.open(image_root + '/' + img_name[n])
        	image = image.convert('RGB')
        	if self.transform is not None:
        		image = self.transform(image)
        	video[n,:,:,:] = image
        	
        label = self.labels[idx]        
        if self.target_transform is not None:
            label = self.target_transform(label)
        return video, label, num_frame


class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def get_ar_data_loader(task, num_per_class=1, split='train',shuffle = False):
    normalize = transforms.Normalize(mean=[0.3474, 0.3474, 0.3474], std=[0.2100, 0.2100, 0.2100])

    dataset = ARDataset(task,split=split,transform=transforms.Compose([transforms.Resize([128,128]), transforms.ToTensor(), normalize]))

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num,shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num,shuffle=shuffle)

    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)

    return loader


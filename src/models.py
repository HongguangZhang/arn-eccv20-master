import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureEncoder3D(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(FeatureEncoder3D, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv3d(3,64,kernel_size=3,padding=1),
                        nn.BatchNorm3d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool3d(2)) # frame/2 x 64 x 64
        self.layer2 = nn.Sequential(
                        nn.Conv3d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm3d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool3d((2,2,2))) # frame/2 x 32 x 32
        self.layer3 = nn.Sequential(
                        nn.Conv3d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm3d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv3d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm3d(64, momentum=1, affine=True),
                        nn.ReLU()) # frame/2 x 32 x 32

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

class TemporalDetector(nn.Module):
    """docstring for ClassName"""
    def __init__(self, F):
        super(TemporalDetector, self).__init__()
       	self.td1 = nn.Sequential(
                        nn.Conv3d(F,F,kernel_size=3,padding=1),
                        nn.ReLU(),
                        nn.MaxPool3d(2), # frame/2 x 32 x 32 x 32
                        nn.Conv3d(F,F,kernel_size=3,padding=1),
                        nn.ReLU(),
                        nn.MaxPool3d((32,16,16)),
                        nn.Conv3d(F,F,kernel_size=1,padding=0),
                        nn.Sigmoid()) # frame/2 x 1 x 1 x 1                    
    def forward(self,x):
        td = self.td1(x.permute(0,2,1,3,4))
        return td.permute(0,2,1,3,4)

class TemporalDetectorV1(nn.Module):
    """docstring for ClassName"""
    def __init__(self, F):
        super(TemporalDetectorV1, self).__init__()
       	self.td1 = nn.Sequential(
                        nn.Conv3d(64,16,kernel_size=3,padding=1),
                        nn.BatchNorm3d(16, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool3d((1,4,4)), # frame/2 x 32 x 32 x 32
                        nn.Conv3d(16,16,kernel_size=3,padding=1),
                        nn.BatchNorm3d(16, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool3d((1,8,8)),
                        nn.Conv3d(16,1,kernel_size=1,padding=0),
                        nn.Sigmoid()) # frame/2 x 1 x 1 x 1                    
    def forward(self,x):
        td = self.td1(x)
        return td

class VanillaTemporalDetector(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(VanillaTemporalDetector, self).__init__()
       	self.td1 = nn.Sequential(
                        nn.Conv3d(64,16,kernel_size=3,padding=1),
                        nn.BatchNorm3d(16, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool3d((1,4,4)), # frame/2 x 32 x 32 x 32
                        nn.Conv3d(16,16,kernel_size=3,padding=1),
                        nn.BatchNorm3d(16, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool3d((1,8,8)),
                        nn.Conv3d(16,1,kernel_size=1,padding=0)) # frame/2 x 1 x 1 x 1                    
    def forward(self,x):
        td = self.td1(x)
        return td
        
class SoftmaxTemporalDetector(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(SoftmaxTemporalDetector, self).__init__()
       	self.td1 = nn.Sequential(
                        nn.Conv3d(64,16,kernel_size=3,padding=1),
                        nn.BatchNorm3d(16, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool3d((1,4,4)), # frame/2 x 32 x 32 x 32
                        nn.Conv3d(16,16,kernel_size=3,padding=1),
                        nn.BatchNorm3d(16, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool3d((1,8,8)),
                        nn.Conv3d(16,1,kernel_size=1,padding=0),
                        nn.Softmax(dim=2)) # frame/2 x 1 x 1 x 1                    
    def forward(self,x):
        td = self.td1(x)
        return td
        
class SpatialDetector(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(SpatialDetector, self).__init__()
       	self.sd1 = nn.Sequential(
                        nn.Conv3d(64,16,kernel_size=3,padding=1),
                        nn.BatchNorm3d(16, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool3d((2,1,1)), # frame/2 x 32 x 32 x 32
                        nn.Conv3d(16,16,kernel_size=3,padding=1),
                        nn.BatchNorm3d(16, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool3d((2,1,1)),
                        nn.Conv3d(16,1,kernel_size=1,padding=0),
                        nn.Sigmoid()) # 1 x 1 x H x W                   
    def forward(self,x):
        sd = self.sd1(x)
        return sd

class VanillaSpatialDetector(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(VanillaSpatialDetector, self).__init__()
       	self.sd1 = nn.Sequential(
                        nn.Conv3d(64,16,kernel_size=3,padding=1),
                        nn.BatchNorm3d(16, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool3d((2,1,1)), # frame/2 x 32 x 32 x 32
                        nn.Conv3d(16,16,kernel_size=3,padding=1),
                        nn.BatchNorm3d(16, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool3d((2,1,1)),
                        nn.Conv3d(16,1,kernel_size=1,padding=0)) # 1 x 1 x H x W                   
    def forward(self,x):
        sd = self.sd1(x)
        return sd
        
class SoftmaxSpatialDetector(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(SoftmaxSpatialDetector, self).__init__()
       	self.sd1 = nn.Sequential(
                        nn.Conv3d(64,16,kernel_size=3,padding=1),
                        nn.BatchNorm3d(16, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool3d((2,1,1)), # frame/2 x 32 x 32 x 32
                        nn.Conv3d(16,16,kernel_size=3,padding=1),
                        nn.BatchNorm3d(16, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool3d((2,1,1)),
                        nn.Conv3d(16,1,kernel_size=1,padding=0))
       	self.sf = nn.Softmax2d() # 1 x 1 x H x W                   
    def forward(self,x):
        sd = self.sd1(x)
        b = sd.size(0)
        t = sd.size(1)
        h = sd.size(3)
        w = sd.size(4)
        sd = self.sf(sd.view(-1,1,sd.size(3),sd.size(4))).view(b,t,1,h,w)
        return sd
                        
class FeatureEncoderP3D(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(FeatureEncoderP3D, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv3d(3,64,kernel_size=[1,3,3],padding=[0,1,1]),
                        nn.BatchNorm3d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv3d(64,64,kernel_size=[3,1,1],padding=[1,0,0]),
                        nn.BatchNorm3d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool3d(2)) # frame/2 x 64 x 64
        self.layer2 = nn.Sequential(
                        nn.Conv3d(64,64,kernel_size=[1,3,3],padding=[0,1,1]),
                        nn.BatchNorm3d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv3d(64,64,kernel_size=[3,1,1],padding=[1,0,0]),
                        nn.BatchNorm3d(64, momentum=1, affine=True),
                        nn.ReLU()) # frame/4 x 32 x 32
        self.layer3 = nn.Sequential(
                        nn.Conv3d(64,64,kernel_size=[1,3,3],padding=[0,1,1]),
                        nn.BatchNorm3d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv3d(64,64,kernel_size=[3,1,1],padding=[1,0,0]),
                        nn.BatchNorm3d(64, momentum=1, affine=True),
                        nn.ReLU()) # frame/4 x 16 x 16
        self.layer4 = nn.Sequential(
                        nn.Conv3d(64,64,kernel_size=[1,3,3],padding=[0,1,1]),
                        nn.BatchNorm3d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv3d(64,64,kernel_size=[3,1,1],padding=[1,0,0]),
                        nn.BatchNorm3d(64, momentum=1, affine=True),
                        nn.ReLU()) # frame/4 x 16 x 16

    def forward(self,x):
        out = self.layer1(x)
        out = nn.MaxPool3d(2)(out + self.layer2(out))
        out = out + self.layer3(out)
        out = out + self.layer4(out)
        return out
        
class SimilarityNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(SimilarityNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(2,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU()) #Nx64x32x32
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU()) #Nx64x16x16
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU()) #Nx64x8x8
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU()) #Nx64x4x4           
        self.fc1 = nn.Linear(input_size*16,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = nn.MaxPool2d(2)(self.layer1(x))
        out = nn.MaxPool2d(2)(out + self.layer2(out))      
        out = nn.MaxPool2d(2)(out + self.layer3(out))
        out = nn.MaxPool2d(2)(out + self.layer4(out))
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out
        
class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_dim):
        super(RelationNetwork, self).__init__()         
        self.fc1 = nn.Linear(2*input_dim,1024)
        self.fc2 = nn.Linear(1024,256)
        self.fc3 = nn.Linear(256,1)

    def forward(self,x):
        print(x.size())
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        return out        
        
class Discriminator(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(64**2,32**2)
        self.fc2 = nn.Linear(32**2,16**2)
        self.fc3 = nn.Linear(16**2,8**2)
        self.fc4 = nn.Linear(8**2,dim)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out
        
class Discriminator3D(nn.Module):
    """docstring for ClassName"""
    def __init__(self, out_dim):
        super(Discriminator3D, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv3d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm3d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool3d(2)) # 64 x 2 x 16 x 16
        self.layer2 = nn.Sequential(
                        nn.Conv3d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm3d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool3d((2))) # 64 x 1 x 8 x 8
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, out_dim)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F
from lib import *
from config import *

"""
Convolution neural network model
Input_1: PPG + VPG + APG (3000,1)
Input_2: Personal Information + length of original PPG signal before convert to 1000
Output: SBP and DBP
"""
class CNN_PPG_VPG_APG_info(nn.Module):
    def __init__(self):
        super(CNN_PPG_VPG_APG_info,self).__init__()
        self.conv1 = nn.Conv2d(1,8,kernel_size=(10,1), stride=(2,1))
        self.conv2 = nn.Conv2d(8,32,kernel_size=(10,1), stride=(2,1))
        self.conv3 = nn.Conv2d(32,32, kernel_size=(10,1), stride=(2,1))
        self.fc1   = nn.Linear(800,400)
        self.fc2   = nn.Linear(400,32)
        self.fc3   = nn.Linear(37,2)

    def forward(self,x1,x2):
        # x1.shape = (batch_size, 1,3000,1)
        # x2.shape = (batch_size, 1,5)
        x1 = F.max_pool2d(x1,(3,1))  # (batch_size,1,3000,1) -> (batch_size,1,1000,1)

        x1 = self.conv1(x1)          # (batch_size,1,1000,1) -> (batch_size,8,496,1)
        x1 = F.relu(x1)

        x1 = F.max_pool2d(x1,(4,1))  # (batch_size,8,496,1) -> (batch_size,8,124,1)

        x1 = self.conv2(x1)          # (batch_size,8,124,1) -> (batch_size,32,58,1)
        x1 = F.relu(x1)

        x1 = self.conv3(x1)          # (batch_size,32,58,1) -> (batch_size,32,25,1)
        x1 = F.relu(x1)

        x1 = x1.view(-1, 800)       # (batch_size,32,25,1)  -> (batch_size,800)

        x1 = self.fc1(x1)           # (batch_size,800)  ->  (batch_size,400)
        x1 = F.relu(x1)

        x1 = self.fc2(x1)           # (batch_size,400)  ->  (batch_size,32)
        x1 = F.relu(x1)

        x2 = x2.view(-1,5)          # (batch_size, 1,5) ->  (batch_size,5)

        # add personal informatino
        x1 = torch.cat([x1,x2],1)   # (batch_size,32) + (batch_size,5)  -> (batch_size,37)

        x1 = self.fc3(x1)            # (batch_size,37)  ->  (batch_size,2)
        x1 = F.relu(x1)

        return x1     # (batch_size,2)


"""
Convolution neural network model
Input_1: PPG + VPG + APG (3000,1)
Input_2: length of original PPG before convert to 1000
Output: SBP and DBP
"""
class CNN_PPG_VPG_APG(nn.Module):
    def __init__(self):
        super(CNN_PPG_VPG_APG,self).__init__()
        self.conv1 = nn.Conv2d(1,8,kernel_size=(10,1), stride=(2,1))
        self.conv2 = nn.Conv2d(8,32,kernel_size=(10,1), stride=(2,1))
        self.conv3 = nn.Conv2d(32,32, kernel_size=(10,1), stride=(2,1))
        self.fc1   = nn.Linear(800,400)
        self.fc2   = nn.Linear(400,32)
        self.fc3   = nn.Linear(33,2)

    def forward(self,x1,x2):
        # x1.shape = (batch_size, 1,3000,1)
        # x2.shape = (batch_size, 1,1)
        x1 = F.max_pool2d(x1,(3,1))  # (batch_size,1,2000,1) -> (batch_size,1,1000,1)

        x1 = self.conv1(x1)          # (batch_size,1,1000,1) -> (batch_size,8,496,1)
        x1 = F.relu(x1)

        x1 = F.max_pool2d(x1,(4,1))  # (batch_size,8,496,1) -> (batch_size,8,124,1)

        x1 = self.conv2(x1)          # (batch_size,8,124,1) -> (batch_size,32,58,1)
        x1 = F.relu(x1)

        x1 = self.conv3(x1)          # (batch_size,32,58,1) -> (batch_size,32,25,1)
        x1 = F.relu(x1)

        x1 = x1.view(-1, 800)       # (batch_size,32,25,1)  -> (batch_size,800)

        x1 = self.fc1(x1)           # (batch_size,800)  ->  (batch_size,400)
        x1 = F.relu(x1)

        x1 = self.fc2(x1)           # (batch_size,400)  ->  (batch_size,32)
        x1 = F.relu(x1)

        x2 = x2.view(-1,1)          # (batch_size, 1,1) ->  (batch_size,1)

        # add original ppg length
        x1 = torch.cat([x1,x2],1)   # (batch_size,32) + (batch_size,1)  -> (batch_size,33)

        x1 = self.fc3(x1)            # (batch_size,33)  ->  (batch_size,2)
        x1 = F.relu(x1)

        return x1     # (batch_size,2)


"""
Convolution neural network model
Input_1: PPG + VPG (2000,1)
Input_2: Personal Information + length of original PPG signal before convert to 1000
Output: SBP and DBP
"""
class CNN_PPG_VPG(nn.Module):
    def __init__(self):
        super(CNN_PPG_VPG,self).__init__()
        self.conv1 = nn.Conv2d(1,8,kernel_size=(10,1), stride=(2,1))
        self.conv2 = nn.Conv2d(8,32,kernel_size=(10,1), stride=(2,1))
        self.conv3 = nn.Conv2d(32,32, kernel_size=(10,1), stride=(2,1))
        self.fc1   = nn.Linear(800,400)
        self.fc2   = nn.Linear(400,32)
        self.fc3   = nn.Linear(33,2)

    def forward(self,x1,x2):
        # x1.shape = (batch_size, 1,2000,1)
        # x2.shape = (batch_size, 1,1)
        x1 = F.max_pool2d(x1,(2,1))  # (batch_size,1,2000,1) -> (batch_size,1,1000,1)

        x1 = self.conv1(x1)          # (batch_size,1,1000,1) -> (batch_size,8,496,1)
        x1 = F.relu(x1)

        x1 = F.max_pool2d(x1,(4,1))  # (batch_size,8,496,1) -> (batch_size,8,124,1)

        x1 = self.conv2(x1)          # (batch_size,8,124,1) -> (batch_size,32,58,1)
        x1 = F.relu(x1)

        x1 = self.conv3(x1)          # (batch_size,32,58,1) -> (batch_size,32,25,1)
        x1 = F.relu(x1)

        x1 = x1.view(-1, 800)       # (batch_size,32,25,1)  -> (batch_size,800)

        x1 = self.fc1(x1)           # (batch_size,800)  ->  (batch_size,400)
        x1 = F.relu(x1)

        x1 = self.fc2(x1)           # (batch_size,400)  ->  (batch_size,32)
        x1 = F.relu(x1)

        x2 = x2.view(-1,1)          # (batch_size, 1,1) ->  (batch_size,1)

        # add original length of ppg signal after convert to 1000
        x1 = torch.cat([x1,x2],1)   # (batch_size,32) + (batch_size,1)  -> (batch_size,33)

        x1 = self.fc3(x1)            # (batch_size,33)  ->  (batch_size,2)
        x1 = F.relu(x1)

        return x1     # (batch_size,2)

"""
Convolution neural network model
Input_1: PPG (1000,1)
Input_2: Personal Information + length of original PPG signal before convert to 1000
Output: SBP and DBP
"""
class CNN_PPG(nn.Module):
    def __init__(self):
        super(CNN_PPG,self).__init__()
        self.conv1 = nn.Conv2d(1,8,kernel_size=(10,1), stride=(2,1))
        self.conv2 = nn.Conv2d(8,32,kernel_size=(10,1), stride=(2,1))
        self.conv3 = nn.Conv2d(32,32, kernel_size=(10,1), stride=(2,1))
        self.fc1   = nn.Linear(800,400)
        self.fc2   = nn.Linear(400,32)
        self.fc3   = nn.Linear(33,2)

    def forward(self,x1,x2):
        # x1.shape = (batch_size, 1,1000,1)
        # x2.shape = (batch_size, 1,1)
        # x1 = F.max_pool2d(x1,(1,1))  # (batch_size,1,1000,1) -> (batch_size,1,1000,1)

        x1 = self.conv1(x1)          # (batch_size,1,1000,1) -> (batch_size,8,496,1)
        x1 = F.relu(x1)

        x1 = F.max_pool2d(x1,(4,1))  # (batch_size,8,496,1) -> (batch_size,8,124,1)

        x1 = self.conv2(x1)          # (batch_size,8,124,1) -> (batch_size,32,58,1)
        x1 = F.relu(x1)

        x1 = self.conv3(x1)          # (batch_size,32,58,1) -> (batch_size,32,25,1)
        x1 = F.relu(x1)

        x1 = x1.view(-1, 800)       # (batch_size,32,25,1)  -> (batch_size,800)

        x1 = self.fc1(x1)           # (batch_size,800)  ->  (batch_size,400)
        x1 = F.relu(x1)

        x1 = self.fc2(x1)           # (batch_size,400)  ->  (batch_size,32)
        x1 = F.relu(x1)

        x2 = x2.view(-1,1)          # (batch_size, 1,1) ->  (batch_size,1)

        # add original length of ppg signal after convert to 1000
        x1 = torch.cat([x1,x2],1)   # (batch_size,32) + (batch_size,1)  -> (batch_size,33)

        x1 = self.fc3(x1)            # (batch_size,33)  ->  (batch_size,2)
        x1 = F.relu(x1)

        return x1     # (batch_size,2)


def test_model(option = "CNN_PPG_VPG_APG_info"):
    if option == "CNN_PPG_VPG_APG_info":
        net = CNN_PPG_VPG_APG_info().cuda()
        x1 = torch.randn(3,1,3000,1).to("cuda")
        x2 = torch.randn(3,1,5).to("cuda")
        y = net(x1,x2).to("cuda")
        print(y.size())
        summary(net, [(1, 3000, 1), (1, 1, 5)])
    elif option == "CNN_PPG_VPG_APG":
        net = CNN_PPG_VPG_APG().cuda()
        x1 = torch.randn(3,1,3000,1).to("cuda")
        x2 = torch.randn(3,1,1).to("cuda")
        y = net(x1,x2).to("cuda")
        print(y.size())
        summary(net, [(1, 3000, 1), (1, 1, 1)])
    elif option == "CNN_PPG_VPG":
        net = CNN_PPG_VPG().cuda()
        x1 = torch.randn(3, 1, 2000, 1).to("cuda")
        x2 = torch.randn(3, 1, 1).to("cuda")
        y = net(x1, x2).to("cuda")
        print(y.size())
        summary(net, [(1, 2000, 1), (1, 1, 1)])
    elif option == "CNN_PPG":
        net = CNN_PPG().cuda()
        x1 = torch.randn(3, 1, 1000, 1).to("cuda")
        x2 = torch.randn(3, 1, 1).to("cuda")
        y = net(x1, x2).to("cuda")
        print(y.size())
        summary(net, [(1, 1000, 1), (1, 1, 1)])
    else: return None

def create_model(option):
    if option == model_name[0]:
        model = CNN_PPG()
    elif option == model_name[1]:
        model = CNN_PPG_VPG()
    elif option == model_name[2]:
        model = CNN_PPG_VPG_APG()
    elif option == model_name[3]:
        model = CNN_PPG_VPG_APG_info()
    else:
        model = None
    return model


if __name__ == "__main__":
    test_model(option="CNN_PPG_VPG_APG_info")
    test_model(option="CNN_PPG_VPG_APG")
    test_model(option="CNN_PPG_VPG")
    test_model(option="CNN_PPG")

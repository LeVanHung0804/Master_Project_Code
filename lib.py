import glob  #liệt kê part chứa các bức ảnh
import os
import os.path as osp
import random # dùng trong hàm transform
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
# %matplotlib inline

import pandas as pd
from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data # chứa các hàm số để điều khiển data
import torchvision #chứa hàm số để làm computer vision
from torchvision import models, transforms
from tqdm import tqdm

import argparse

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

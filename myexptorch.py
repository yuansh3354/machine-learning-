###导入需要的包
import os
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

import torch
import torchvision
import torch.utils.data as Data # 数据管理
import torch.nn as nn # 神经网路 
import torch.nn.functional as F # 损失函数
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler,MinMaxScaler

import warnings
from myexptorch import *
#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'
warnings.filterwarnings("ignore")

# 如果有GPU则将数据传递到GPU上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #设置 device 将数据转移到gpu上

"""
这里面存的是常用的公用 pytorch 模型公式
"""

# 将表达谱转换为tensor类型,这里要注意一下,必须要转换为float类型
def expToTorch(x):
    return torch.from_numpy(np.array(x)).float()


# 数据分类的label转换为tensor 这里要注意,一定要转换为longtensor类型
def labelTotorch(y):
    return torch.LongTensor(y)


# 将数据变成迭代器的类型
# 有的数据没有label的时候,可以自己创建一个等长的全为0的label
# 还有要注意一点,batchsize一定要是能被整除的,要么删掉多余的样本,要么上采样进行样本补充   
def makeDataiter(x,y,batch_szie):
    return Data.DataLoader(Data.TensorDataset(x, y), batch_szie, shuffle=False)


def normalize(my_data,std_method):
    if std_method == 'minmax':
        method = MinMaxScaler()
    if std_method == 'std':
        method = StandardScaler()
    
    my_data = pd.DataFrame(method.fit(my_data).transform(
        my_data),columns=my_data.columns,index=my_data.index)
    return my_data
    
























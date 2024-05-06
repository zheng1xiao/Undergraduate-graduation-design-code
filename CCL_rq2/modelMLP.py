import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.autograd import Variable
import numpy as np

# class LogisticRegression1(nn.Module):
#     def __init__(self):
#         super(LogisticRegression1,self).__init__()
#         self.lr=nn.Linear(14,1)   #相当于通过线性变换y=x*T(A)+b可以得到对应的各个系数
#         self.sm=nn.Sigmoid()   #相当于通过激活函数的变换
#
#     def forward(self, x):
#         x=self.lr(x)
#         x=self.sm(x)
#         return x

class MLP(nn.Module):

    def __init__(self, num_i, num_h1, num_h2, num_o):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(num_i, num_h1)
        self.lkrelu1 = nn.LeakyReLU(0.05)
        self.linear2 = nn.Linear(num_h1, num_h2)  # 2个隐层
        self.lkrelu2 = nn.LeakyReLU(0.05)
        self.linear3 = nn.Linear(num_h2, num_o)
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = self.linear1(x)
        x = self.lkrelu1(x)
        x = self.linear2(x)
        x = self.lkrelu2(x)
        x = self.linear3(x)
        return x



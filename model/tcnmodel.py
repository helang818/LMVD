from collections import OrderedDict
import torch.nn as nn
import torch
import numpy as np
import logging
from torch.autograd import Function
from math import sqrt
import torch
import torch.nn.functional as F

class Conv1d(nn.Module):
    def __init__(self) -> None:
        super(Conv1d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(171, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class AstroModel(nn.Module):
    def __init__(self) -> None:
        super(AstroModel, self).__init__()
        self.conv = nn.Conv1d(128, 256, 1)
        self.dropout = nn.Dropout(0.2)

        self.conv1_1 = nn.Conv1d(128, 128, 3, padding=2, dilation=2)
        self.conv1_2 = nn.Conv1d(128, 128, 3, padding=2, dilation=2)

        self.conv2_1 = nn.Conv1d(128, 128, 3, padding=4, dilation=4)
        self.conv2_2 = nn.Conv1d(128, 128, 3, padding=4, dilation=4)

        self.conv3_1 = nn.Conv1d(128, 128, 3, padding=8, dilation=8)
        self.conv3_2 = nn.Conv1d(128, 128, 3, padding=8, dilation=8)

        self.conv4_1 = nn.Conv1d(128, 256, 3, padding=16, dilation=16)
        self.conv4_2 = nn.Conv1d(256, 256, 3, padding=16, dilation=16)
    
    def forward(self, x):
        raw = x
        x = F.relu(self.conv1_1(x))
        x = self.dropout(x)
        x = self.dropout(self.conv1_2(x))
        raw = F.relu(x + raw)

        x = raw
        x = F.relu(self.conv2_1(x))
        x = self.dropout(x)
        x = self.dropout(self.conv2_2(x))
        raw = F.relu(x + raw)

        x = raw
        x = F.relu(self.conv3_1(x))
        x = self.dropout(x)
        x = self.dropout(self.conv3_2(x))
        raw = F.relu(x + raw)

        x = raw
        x = F.relu(self.conv4_1(x))
        x = self.dropout(x)
        x = self.dropout(self.conv4_2(x))
        raw = self.conv(raw)
        raw = F.relu(x + raw)

        return raw
        
class TCNModel(nn.Module):
    def __init__(self) -> None:
        super(TCNModel, self).__init__()
        self.Conv1d = Conv1d()
        self.AstroModel = AstroModel()

    def forward(self, input):
        input = input.transpose(1,2)
        x = self.Conv1d(input)
        x = self.AstroModel(x)
        x = x.transpose(1,2)
        return x

class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Qx, Kx, Vx,Qy,Ky,Vy):
        attentionx = torch.matmul(Qx, torch.transpose(Kx, -1, -2))
        attentiony = torch.matmul(Qy, torch.transpose(Ky, -1, -2))
        attention = torch.cat((attentionx,attentiony),dim=1)
        B,C,H,W = attention.size()
        attention = attention.reshape(B,2,C//2,H,W)
        attention = torch.mean(attention,dim=1).squeeze()
        attention1= torch.softmax(attention / sqrt(Qx.size(-1)), dim=-1)
        attention1 = torch.matmul(attention1, Vx)
        attention2 = torch.softmax(attention / sqrt(Qx.size(-1)), dim=-1)
        attention2 = torch.matmul(attention2, Vy)
        return attention1,attention2

class FeedForward(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out=None, *, dropout=0.0, f=nn.Conv1d, activation=nn.ELU):
        super(FeedForward, self).__init__()
        dim_out = dim_in if dim_out is None else dim_out

        self.net = nn.Sequential(
            f(in_channels=dim_in, out_channels=hidden_dim,kernel_size=1,padding=0,stride=1),
            activation(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            f(in_channels=hidden_dim, out_channels=dim_out,kernel_size=1,padding=0,stride=1),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Multi_CrossAttention(nn.Module):
    def __init__(self, hidden_size, all_head_size, head_num):
        super().__init__()
        self.hidden_size = hidden_size
        self.all_head_size = all_head_size
        self.num_heads = head_num
        self.h_size = all_head_size // head_num
        assert all_head_size % head_num == 0
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.norm = sqrt(all_head_size)
    def print(self):
        print(self.hidden_size, self.all_head_size)
        print(self.linear_k, self.linear_q, self.linear_v)
    def forward(self, x,y):
        batch_size = x.size(0)
        q_sx = self.linear_q(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        k_sx = self.linear_k(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        v_sx = self.linear_v(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        q_sy = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        k_sy = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        v_sy = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)


        attention1,attention2 = CalculateAttention()(q_sx, k_sx, v_sx,q_sy,k_sy,v_sy)
        attention1 = attention1.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)+x
        attention2 = attention2.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)+y

        return attention1,attention2

class ConvNet1d(nn.Module):
    def __init__(self) -> None:
        super(ConvNet1d, self).__init__()
        self.fc = nn.Linear(256, 128)
    
    def forward(self,input):
        sizeTmp = input.size(1)
        batch_size = input.size(0)
        outConv1d = input.contiguous().view(input.size(0)*input.size(1),-1)
        output = self.fc(outConv1d)
        output = output.view(batch_size, sizeTmp, -1)

        return output



class gateRegress():
    def __init__(self) -> None:
        pass
    def forward(self, ):
        pass

class Regress2(nn.Module):
    def __init__(self) -> None:
        super(Regress2, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(186, 64),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 2),
            nn.ELU())

            
    def forward(self, x):
        x = x.view(-1, 186)
        x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.TCNModel = TCNModel() 
        self.Conv1dModel = ConvNet1d()
        self.Regress = Regress2()

        self.softmax = torch.nn.Softmax(dim=1)
        self.conv = nn.Conv1d(in_channels=114,out_channels=186,kernel_size=1,padding=0,stride=1)
        self.mhca = Multi_CrossAttention(hidden_size=128,all_head_size=128,head_num=4)
        self.norm = nn.LayerNorm(128*2)
        self.FFN = FeedForward(dim_in=186, hidden_dim=186*2,dim_out =186)
        self.norm2 = nn.LayerNorm(128*2)
        self.pooling = nn.AdaptiveAvgPool1d(1)
    def forward(self,inputVideo,inputAudio): 

        inputVideo = self.TCNModel(inputVideo)

        outputConv1dVideo = self.Conv1dModel(inputVideo)

        outputConv1dVideo = self.conv(outputConv1dVideo)

        output1,output2 = self.mhca(outputConv1dVideo,inputAudio)
        
        outputFeature = torch.cat((output1, output2),dim=2) 
        outputFeature = self.FFN(self.norm(outputFeature))+outputFeature
        
        output = self.norm2(outputFeature)
        output = self.pooling(output).reshape(output.shape[0],-1)
       

        result = self.Regress(output)
        result = result.squeeze(-1)
        result = self.softmax(result)
        return result

if __name__ == '__main__':

    model = Net().cuda()
    Conv1dModel = ConvNet1d()
    x1 = torch.randn(4,186,128).cuda()
    x2 = torch.randn(4,915,171).cuda()
    y = model(x2,x1)
    print(y.shape)
    


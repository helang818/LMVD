from torch import nn
import torch


class Bottlrneck(torch.nn.Module):
    def __init__(self,In_channel,Med_channel,Out_channel,downsample=False):
        super(Bottlrneck, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Out_channel, 1),
            torch.nn.BatchNorm1d(Out_channel),
        )

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d((1)),
            nn.Conv1d(Out_channel, Out_channel // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(Out_channel // 16, Out_channel, kernel_size=1),
            nn.Sigmoid()
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv1d(In_channel, Out_channel,1,self.stride)
        else:
            self.res_layer = None

    def forward(self,x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        x1 = self.layer(x)
        x2 = self.se(x1)
        x1 = x2*x1
        return x1+residual


class ResNet(torch.nn.Module):
    def __init__(self,in_channels=2,classes=5):
        super(ResNet, self).__init__()
        self.linear = nn.Linear(171, 128)
        self.conv = nn.Conv1d(in_channels=915, out_channels=186, kernel_size=1, stride=1)
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels,64,kernel_size=7,stride=2,padding=3),
            torch.nn.MaxPool1d(3,2,1),

            Bottlrneck(64,64,256,False),
            Bottlrneck(256,64,256,False),
            Bottlrneck(256,64,256,False),
            #
            Bottlrneck(256,128,512, True),
            Bottlrneck(512,128,512, False),
            Bottlrneck(512,128,512, False),
            Bottlrneck(512,128,512, False),
            #
            Bottlrneck(512,256,1024, True),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            #
            Bottlrneck(1024,512,2048, True),
            Bottlrneck(2048,512,2048, False),
            Bottlrneck(2048,512,2048, False),

            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(2048,classes)
        )

    def forward(self,X1,X2):
        X1 = self.conv(X1)
        X1 = self.linear(X1)
        #X = X1 + X2
        X = torch.cat([X1,X2],dim=2)
        x = self.features(X)
        x = x.view(-1,2048)
        x = self.classifer(x)
        return x


if __name__ == '__main__':
    model = ResNet(in_channels=186,classes=2)
    x = torch.randn(4,915,171)
    y = torch.randn(4,186,128)

    out = model(x,y)
    print(model)
    print(out.shape)
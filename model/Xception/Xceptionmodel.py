import torch

class SeparableConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(SeparableConv1d, self).__init__()


        self.depthwise = torch.nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)

        self.pointwise = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Entry(torch.nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.beforeresidual = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels,32,3,2,1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, 3, 2, 1),
            torch.nn.ReLU()
        )

        self.residual_branch1 = torch.nn.Conv1d(64, 128, 1, 2)
        self.residual_model1 = torch.nn.Sequential(
            SeparableConv1d(64,128,3,1,1),
            torch.nn.ReLU(),
            SeparableConv1d(128, 128, 3, 1, 1),
            torch.nn.MaxPool1d(3,2,1)
        )

        self.residual_branch2 = torch.nn.Conv1d(256, 256, 1, 2)
        self.residual_model2 = torch.nn.Sequential(
            torch.nn.ReLU(),
            SeparableConv1d(256,256,3,1,1),
            torch.nn.ReLU(),
            SeparableConv1d(256, 256, 3, 1, 1),
            torch.nn.MaxPool1d(3,2,1)
        )

        self.residual_branch3 = torch.nn.Conv1d(512, 728, 1, 2)
        self.residual_model3 = torch.nn.Sequential(
            torch.nn.ReLU(),
            SeparableConv1d(512,728,3,1,1),
            torch.nn.ReLU(),
            SeparableConv1d(728, 728, 3, 1, 1),
            torch.nn.MaxPool1d(3,2,1)
        )


    def forward(self,x):
        x = self.beforeresidual(x)

        x1 = self.residual_branch1(x)
        x = self.residual_model1(x)
        x = torch.cat([x,x1],dim=1)

        x1 = self.residual_branch2(x)
        x = self.residual_model2(x)
        x = torch.cat([x,x1],dim=1)

        x1 = self.residual_branch3(x)
        x = self.residual_model3(x)
        x = x+x1

        return x


class Middleflow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.ReLU(),
            SeparableConv1d(728,728,3,1,1),
            torch.nn.ReLU(),
            SeparableConv1d(728, 728, 3, 1, 1),
            torch.nn.ReLU(),
            SeparableConv1d(728, 728, 3, 1, 1),
        )

    def forward(self,x):
        return x + self.layers(x)


class Exitflow(torch.nn.Module):
    def __init__(self,classes):
        super().__init__()

        self.residual = torch.nn.Conv1d(728,1024,1,2)
        self.residual_model = torch.nn.Sequential(
            torch.nn.ReLU(),
            SeparableConv1d(728,728,3,1,1),
            torch.nn.ReLU(),
            SeparableConv1d(728, 1024, 3, 1, 1),
            torch.nn.MaxPool1d(3,2,1)
        )
        self.last_layer = torch.nn.Sequential(
            SeparableConv1d(1024,1536,3,1,1),
            torch.nn.ReLU(),
            SeparableConv1d(1536, 2048, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(2048,classes)
        )


    def forward(self,x):
        x = self.residual_model(x) + self.residual(x)
        x = self.last_layer(x)

        return x



class Xception(torch.nn.Module):
    def __init__(self,in_channels,classes):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels=915, out_channels=186, kernel_size=1, stride=1)
        self.linear = torch.nn.Linear(in_features=171, out_features=128)
        self.layers = torch.nn.Sequential(
            Entry(in_channels),
            Middleflow(),
            Middleflow(),
            Middleflow(),
            Middleflow(),
            Middleflow(),
            Middleflow(),
            Middleflow(),
            Middleflow(),
            Exitflow(classes)
        )
    def forward(self,X1,X2):
        X1 = self.conv(X1)
        X1 = self.linear(X1)
        #X = X1 + X2
        X = torch.cat([X1,X2],dim=2)
        return self.layers(X)


if __name__ == '__main__':
    model = Xception(in_channels=186,classes=2).cuda()
    x = torch.randn(4,915,171).cuda()
    y = torch.randn(4,186,128).cuda()
    out = model(x,y)
    print(model)
    print(out.shape)
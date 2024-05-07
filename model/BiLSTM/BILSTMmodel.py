import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BiLSTM(nn.Module):
    def __init__(self,input,n_class,n_hidden):
        super(BiLSTM, self,).__init__()
        self.n_class =n_class
        self.n_hidden = n_hidden
        self.linear = nn.Linear(171,128)
        self.conv  = nn.Conv1d(in_channels=915,out_channels=186,kernel_size=1,stride=1)
        self.lstm = nn.LSTM(input_size=input, hidden_size=n_hidden, bidirectional=True)
        self.fc = nn.Linear(n_hidden * 2, n_class)

    def forward(self, X1 ,X2):
        batch_size = X1.shape[0]
        X1 = self.conv(X1)
        X1 = self.linear(X1)
        #X = X1+X2
        X = torch.cat([X1,X2],dim=2)
        input = X.transpose(0, 1)

        hidden_state = torch.randn(1 * 2, batch_size,
                                   self.n_hidden).cuda(1)
        cell_state = torch.randn(1 * 2, batch_size,
                                 self.n_hidden).cuda(1)

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]
        model = self.fc(outputs)
        return model


if __name__ == '__main__':
    x1 = torch.randn(4,915,171).cuda(1)
    x2 = torch.randn(4,186,128).cuda(1)
    model = BiLSTM(input = 128,n_class=2,n_hidden=256).cuda(1)
    print(model)
    y = model(x1,x2)
    print(y.shape)
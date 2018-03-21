import torch as th
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class IndRNNCell(nn.Module):

    def __init__(self, inputDim, hiddenDim, nonlinearity="relu"):

        super().__init__()
        # self.W = nn.Parameter(th.randn((inputDim, hiddenDim)))
        # self.b = nn.Parameter(th.randn(hiddenDim))
        self.i2h = nn.Linear(inputDim, hiddenDim, bias=True)
        self.u = nn.Parameter(th.diag(th.randn(hiddenDim)))
        self.act = F.relu if nonlinearity =="relu" else nonlinearity
        self.BN1 = nn.BatchNorm1d(hiddenDim)
        self.BN2 = nn.BatchNorm1d(hiddenDim)

    def forward(self, input, hidden):
        weightOutput = self.i2h(input)
        bnOutput = self.BN1(weightOutput)
        recurrentOutpt = bnOutput + hidden @ self.u # 这里用对角化矩阵相乘模拟hadamard积
        activityOutput = self.act(recurrentOutpt)
        return self.BN2(activityOutput)


class IndRNNModel(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum):

        super().__init__()

        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = layerNum
        self.IndRNNCell = IndRNNCell(inputDim, hiddenNum)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim)

    def forward(self, x, batchSize):

        h0 = Variable(th.zeros(batchSize, self.hiddenNum))
        h_current = h0
        h_last = h0
        inputLen = x.data.size()[1]

        for i in range(inputLen):
            current_input = x[:, i, :]
            h_current = self.IndRNNCell(current_input, h_last)

        fcOutput = self.fc(h_current)

        return nn.Softmax()(fcOutput)



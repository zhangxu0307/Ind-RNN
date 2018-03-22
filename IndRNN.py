import torch as th
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class IndRNNCell(nn.Module):

    def __init__(self, inputDim, hiddenDim, middle=False, nonlinearity=None):

        super().__init__()
        # self.W = nn.Parameter(th.randn((inputDim, hiddenDim)))
        # self.b = nn.Parameter(th.randn(hiddenDim))
        if middle:
            self.i2h = nn.Linear(hiddenDim, hiddenDim, bias=True) # 如果是中间的cell，输入权重不同
        else:
            self.i2h = nn.Linear(inputDim, hiddenDim, bias=True)
        self.u = nn.Parameter(th.diag(th.randn(hiddenDim)))
        self.act = F.relu if nonlinearity == None else nonlinearity
        self.BN1 = nn.BatchNorm1d(hiddenDim)
        self.BN2 = nn.BatchNorm1d(hiddenDim)

    def forward(self, input, hidden):
        weightOutput = self.i2h(input)
        bnOutput = self.BN1(weightOutput)
        recurrentOutpt = bnOutput + hidden @ self.u # 这里用对角化矩阵相乘模拟hadamard积
        activityOutput = self.act(recurrentOutpt)
        return self.BN2(activityOutput)

# class ResidualIndeRNNCell(nn.Module):
#
#     def __init__(self, inputDim, hiddenDim, nonlinearity=None):
#
#         super().__init__()
#
#         self.i2h1 = nn.Linear(inputDim, hiddenDim, bias=True)
#         self.u = nn.Parameter(th.diag(th.randn(hiddenDim)))
#         self.act = F.relu if nonlinearity == None else nonlinearity
#         self.BN1 = nn.BatchNorm1d(hiddenDim)
#         self.BN2 = nn.BatchNorm1d(hiddenDim)
#
#     def forward(self, input, hidden):
#         pass



class IndRNNModel(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum):

        super().__init__()

        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = layerNum
        self.IndRNNCellList = [IndRNNCell(inputDim, hiddenNum, middle=False)]
        self.IndRNNCellList += [IndRNNCell(inputDim, hiddenNum, middle=True)]*(self.layerNum-1)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim)

    def forward(self, x, batchSize):

        # h0 = Variable(th.zeros(batchSize, self.hiddenNum))
        # h_current = h0
        # h_last = h0
        # inputLen = x.data.size()[1]
        #
        # for i in range(inputLen):
        #     current_input = x[:, i, :]
        #     h_current = self.IndRNNCell(current_input, h_last)
        #
        # fcOutput = self.fc(h_current)
        #
        # return nn.Softmax()(fcOutput)

        h0List = [Variable(th.zeros(batchSize, self.hiddenNum))]*self.layerNum
        hCurrentList = h0List
        hLastList = h0List

        inputLen = x.data.size()[1]

        for i in range(inputLen): # 多层layer
            current_input = x[:, i, :]
            for j in range(self.layerNum):
                hCurrentList[j] = self.IndRNNCellList[j](current_input, hLastList[j])
                current_input = hCurrentList[j]

        fcOutput = self.fc(hLastList[-1])

        return nn.Softmax()(fcOutput)



import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from IndRNN2 import *

torch.manual_seed(1)

# 模型基类，主要是用于指定参数和cell类型
class BaseModel(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell):

        super(BaseModel, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = layerNum
        if cell == "RNN":
            self.cell = nn.RNN(input_size=self.inputDim, hidden_size=self.hiddenNum,
                        num_layers=self.layerNum, dropout=0.0,
                         nonlinearity="tanh", batch_first=True,)
        if cell == "LSTM":
            self.cell = nn.LSTM(input_size=self.inputDim, hidden_size=self.hiddenNum,
                               num_layers=self.layerNum, dropout=0.0,
                               batch_first=True, )
        if cell == "GRU":
            self.cell = nn.GRU(input_size=self.inputDim, hidden_size=self.hiddenNum,
                                num_layers=self.layerNum, dropout=0.0,
                                 batch_first=True, )
        if cell == "INDRNN":
            self.cell = IndRNN(input_size=self.inputDim, hidden_size=self.hiddenNum, n_layer= self.layerNum)
        print("cell type:", self.cell)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim)


# 标准RNN模型
class RNNModel(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum):

        super(RNNModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell="RNN")


    def forward(self, x, batchSize):

        rnnOutput, hn = self.cell(x,)

        rnnOutput = rnnOutput[:, -1, :].squeeze()

        fcOutput = self.fc(rnnOutput)
        out = F.log_softmax(fcOutput)

        return out


# LSTM模型
class LSTMModel(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum):

        super(LSTMModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell="LSTM")

    def forward(self, x, batchSize):

        rnnOutput, hn = self.cell(x,)

        rnnOutput = rnnOutput[:, -1, :].squeeze()

        fcOutput = self.fc(rnnOutput)
        out = F.log_softmax(fcOutput)

        return out

# GRU模型
class GRUModel(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum):
        super(GRUModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell="GRU")

    def forward(self, x, batchSize):

        rnnOutput, hn = self.cell(x)
        rnnOutput = rnnOutput[:, -1, :].squeeze()

        out = self.fc(rnnOutput)
        out = F.log_softmax(out)

        return out

class IndRNNModel(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum):
        super(IndRNNModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell="INDRNN")

    def forward(self, x, batchSize):

        #h0 = torch.zeros(self.layerNum*1, batchSize, 28)

        rnnOutput = self.cell(x,)

        rnnOutput = rnnOutput[:, -1, :].squeeze()

        out = self.fc(rnnOutput)
        out = F.log_softmax(out)

        return out

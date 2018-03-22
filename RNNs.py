import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

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
        print("cell type:", self.cell)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim)


# 标准RNN模型
class RNNModel(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum):

        super(RNNModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell="RNN")

    # def init_hidden(self, batchSize):
    #
    #     return hidden

    def forward(self, x, batchSize):

        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        rnnOutput, hn = self.cell(x, h0)
        hn = hn.view(self.layerNum, batchSize, self.hiddenNum) # layernum, batch, hidden
        lastHidden = hn[-1, :, :].view(batchSize, self.hiddenNum)  # 最后一个状态的最后一层
        fcOutput = self.fc(lastHidden)
        out = nn.Softmax()(fcOutput)

        return out


# LSTM模型
class LSTMModel(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum):
        super(LSTMModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell="LSTM")


    def forward(self, x, batchSize):

        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        c0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        rnnOutput, hn = self.cell(x, (h0, c0))  # rnnOutput 12,20,50 hn 1,20,50
        hn = hn[0].view(self.layerNum, batchSize, self.hiddenNum)
        lastHidden = hn[-1, :, :].view(batchSize, self.hiddenNum)
        fcOutput = self.fc(lastHidden)
        out = nn.Softmax()(fcOutput)

        return out

# GRU模型
class GRUModel(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum):
        super(GRUModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell="GRU")

    def forward(self, x, batchSize):

        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        rnnOutput, hn = self.cell(x, h0)  # rnnOutput 12,20,50 hn 1,20,50
        hn = hn.view(self.layerNum, batchSize, self.hiddenNum)
        lastHidden = hn[-1, :, :].view(batchSize, self.hiddenNum)
        fcOutput = self.fc(lastHidden)
        out = nn.Softmax()(fcOutput)

        return out
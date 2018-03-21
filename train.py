import torch as th
import torchvision
from torch.autograd import Variable
from torch import nn
from torch import optim
from torchvision import datasets
import torchvision.transforms as transforms
from sequential_mnist import loadSequentialMNIST
from IndRNN import IndRNNModel


def train(model, batchSize, epoch, useCuda = False):

    if useCuda:
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    ceriation = nn.CrossEntropyLoss()
    trainLoader, testLoader = loadSequentialMNIST(batchSize=batchSize)

    for i in range(epoch):

        # trainning
        ave_loss = 0
        for batch_idx, (x, target) in enumerate(trainLoader):
            optimizer.zero_grad()
            if useCuda:
                x, target = x.cuda(), target.cuda()
            x, target = Variable(x), Variable(target)
            out = model(x, batchSize)
            print(out.size())
            loss = ceriation(out, target)
            ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(trainLoader):
                print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format( i, batch_idx + 1, ave_loss))

        # testing
        correct_cnt, ave_loss = 0, 0
        total_cnt = 0
        for batch_idx, (x, target) in enumerate(testLoader):
            if useCuda:
                x, targe = x.cuda(), target.cuda()
            x, target = Variable(x, volatile=True), Variable(target, volatile=True)
            out = model(x, batchSize)
            loss = ceriation(out, target)
            _, pred_label = th.max(out.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()
            # smooth average
            ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(testLoader):
                print('==>>> epoch: {}, batch index: {}, test loss: {:.6f}, acc: {:.3f}'.format(
                    i, batch_idx + 1, ave_loss, correct_cnt * 1.0 / total_cnt))

    th.save(model.state_dict(), model.name())

if __name__ == '__main__':

    epoch = 10
    batchSize = 128
    model = IndRNNModel(inputDim=1, hiddenNum=256, outputDim=10, layerNum=1)
    train(model, batchSize, epoch, useCuda=False)
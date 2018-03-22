import torch as th
import torchvision
from torch.autograd import Variable
from torchvision import datasets
import torchvision.transforms as transforms


def loadSequentialMNIST(batchSize):

    root = "./data/"

    trans = transforms.Compose(
                 [torchvision.transforms.ToTensor(),
                  torchvision.transforms.Lambda(lambda x: x.view(-1,1))
                 ])
    train_set = datasets.MNIST(root=root, train=True, transform=trans, download=True)
    test_set = datasets.MNIST(root=root, train=False, transform=trans)

    train_loader = th.utils.data.DataLoader(dataset=train_set, batch_size=batchSize, shuffle=True, drop_last=True)
    test_loader = th.utils.data.DataLoader(dataset=test_set, batch_size=batchSize, shuffle=False, drop_last=True)

    print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print ('==>>> total testing batch number: {}'.format(len(test_loader)))

    return train_loader, test_loader

if __name__ == '__main__':

    trainLoader, testLoader = loadSequentialMNIST(batchSize=128)

    for batch_idx, (x, target) in enumerate(trainLoader):
        print(x.size())
        print(target.size())





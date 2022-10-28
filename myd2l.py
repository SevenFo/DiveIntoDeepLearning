import time

import IPython.display
import matplotlib_inline.backend_inline
import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt


# trans = transforms.ToTensor()  # create a transformer to trans data from PIL TO TENSOR(FLOAT 32)
# mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
# mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)

def use_svg_display():  # @save
    """使用svg格式在Jupyter中显示绘图"""
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):  # @save
    """设置matplotlib的图表大小"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


# @save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


# @save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def load_array(data_arrays, batch_size, is_train=True):  # @save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签。"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def get_dataloader_workers():
    return 8


#
# train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=get_dataloader_workers())
#

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)

    return (
        data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers(), drop_last=True),
        data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers(), drop_last=True)
    )


class Accumulator:
    """Accumulator for multiple elements"""

    def __init__(self, n):
        self.data = [0.0] * n  # init n data

    def add(self, *args):
        """add n args at once"""
        assert len(self.data) == len(args), f'length of agrs:{len(args)} is not equal to data length:{len(self.data)}'
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def accuracy(y_hat, y):
    '''
    to calculate the correct quantity in a batch 0~batch_size
    :params y_hat: the possibilities of every type in examples in a batch
    :params y: the index of correct type for a batch, it can be a tensor with size:(batch_size(height),1(width))
    :return: the correct quantity in a batch 0~batch_size
    '''
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)  # an index list
        # the max of a cow (compressed by columns) => index or type(in this case)
        # or get the prediction index of correct type for each picture
        cmp = y_hat.type(y.dtype) == y  # a right index list
        # it may out a vector like [f,t,t,t,t,f,f,t]
        return float(cmp.type(y.dtype).sum())


def accurancy_loss(net, data_iter, loss_function):
    """
    to calculate the loss in a given data_iter, net and loss function
    :param net:
    :param data_iter:
    :param loss_function:
    :return:
    """
    metric = Accumulator(2)
    for features, labels in data_iter:
        metric.add(loss_function(net(features), labels), 1)
    return metric[0] / metric[1]


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """
    to evaluate the accuracy of a given net
    :param net: net to evaluate
    :param data_iter: data used to evaluate, return a batch data for one iteration
    :param device: can be cpu or cude:0 ...
    :return: accuracy per-sent
    """
    if isinstance(net, nn.Module):
        net.eval()  # set net to eval mode may be can speed up
        if not device:
            device = next(iter(net.parameters())).device

    metric = Accumulator(2)  # 2 item accumulator
    with torch.no_grad():
        for X, y in data_iter:  # a batch a time
            if isinstance(X, list):
                X = [x.to(device) for x in X]  # move to gpu or other device
            else:
                X = X.to(device)
            y = y.to(device)  # move y (correct index list) to gpu or other device
            metric.add(accuracy(net(X), y), y.numel())  # accumulate accuracy and elements number (how many input
            # pictures)
    return metric[0] / metric[1]


def train_ch6_gpu(net, train_iter, test_iter, num_epochs, lr, device):
    """

    :param net:
    :param train_iter:
    :param test_iter:
    :param num_epochs: how many time to train
    :param lr:
    :param device:
    :return:
    """

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)  # init weight for every layer can be inited
    print(f'net will be trained on {device}')
    net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # get SGD optimizer
    loss = nn.CrossEntropyLoss()
    num_batches = len(train_iter)
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            start_time = time.time()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
                # print(f'L:{l * X.shape[0]}, accuracy:{accuracy(y_hat,y)}, X.shape[0]:{X.shape[0]}')
            end_time = time.time()
            train_l = metric[0] / metric[2]  # ???
            train_accuracy = metric[1] / metric[2]  # per-sent
            if i % 10 == 0:
                print(
                    f'batch_{i}: train_l:{train_l}, train_accuracy:{train_accuracy}, time_count:{end_time - start_time}')

        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'================================')
        print(f'epoch_{epoch}: test_acc:{test_acc}')
        print(f'================================')


if __name__ == "__main__":
    mnist_train_loader, mnist_test_loader = load_data_fashion_mnist(batch_size=10)
    """
    return format (for a batch): [
        [pic_data], pic_data:Tensor: batch_size x channel_number x pic_height x pic width
        [label_data], label_data: Tensor: height: batch_size, width: 1
    ] batch_num x 2(list)
    """
    # for train_data in mnist_test_loader:
    #
    l = nn.MSELoss(reduction='mean')
    l = l(torch.tensor([1, 2, 3], dtype=torch.float16), torch.tensor([1, 2, 2], dtype=torch.float16))
    print(l)

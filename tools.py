import os
import numpy as np
import torch 
from torch import nn
import time 
import random
import numpy as np 
from IPython import display
from matplotlib import pyplot as plt
from d2l import torch as d2l

def try_gpu(i=0):
    """返回可用的cuda设备，如果没有则返回cpu"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def use_svg_display():
    """在Jupyter中使用svg格式绘图"""
    display.set_matplotlib_formats('svg')

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()
    
    def stop(self):
        """Stop the timer and record the time in a list"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average of time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.tiems)

    def cumsum(self):
        """return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

class Accumulator:
    """For accumulating sums over 'n' variables"""
    def __init__(self, n):
        self.data = [0.] * n
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Animator:
    """绘制动画"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                ylim=None, xscale='linear', yscale='linear', 
                fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
        # lambda function
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, 
        xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向figure中添加数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def seq_data_iter_random(pictures, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随即范围包括num_stpes-1
    pictures = pictures[random.randint(0, num_steps - 1):]
    # 由于要考虑标签，序列长度减一
    num_subseqs = (len(pictures) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在迭代抽样的过程中，
    # 来自两个相邻的、随机的、小批量的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return pictures[pos: pos + num_steps]
    
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # initial_indices 包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [pictures[j + num_steps][0, :, :] for j in initial_indices_per_batch]
        yield torch.tensor(np.array(X), dtype=torch.float), torch.tensor(np.array(Y), dtype=torch.float)

def seq_data_iter_random_repeated(pictures, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列，存在数据的重复使用"""
    # 子序列的起点为所有点
    num_subseqs = len(pictures) - num_steps
    initial_indices = list(range(0, num_subseqs))
    random.shuffle(initial_indices)

    def data(pos):
        return pictures[pos: pos + num_steps]
    
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [pictures[j + num_steps][0, :, :] for j in initial_indices_per_batch]
        yield torch.tensor(np.array(X), dtype=torch.float), torch.tensor(np.array(Y), dtype=torch.float)

def seq_data_iter_sequential(pictures, batch_size, num_steps):
    _, c, h, w = pictures.shape
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_pictures = ((len(pictures) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(np.array(pictures[offset: offset + num_pictures]), dtype=torch.float)
    Xs = Xs.reshape(batch_size, -1, c, h, w)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps, :, :, :]
        Y = Xs[:, i + num_steps, 0, :, :]
        yield X, Y

class SeqDataLoader:
    """An iterator to load sequence data"""
    def __init__(self, pictures, batch_size, num_steps, use_random_iter, path):
        self.pictures = pictures
        self.batch_size = batch_size
        self.num_steps = num_steps

        if use_random_iter:
            self.data_iter = seq_data_iter_random_repeated
        else:
            self.data_iter = seq_data_iter_sequential
        
        if path is not None:
            with open(path + "\\log.txt","a") as f:
                f.write("\n--------TRAIN_DATALOADER--------\n")
                f.write("batch_size: \t\t"+str(self.batch_size)+"\n")
                f.write("num_steps: \t\t\t"+str(self.num_steps)+"\n")
                f.write("use_random_iter: \t"+str(use_random_iter)+"\n")

    def __iter__(self):
        return self.data_iter(self.pictures, self.batch_size, self.num_steps)

def predict(net, data, device):
    raise NotImplementedError()
    data = data.to(device)
    return net(data)

def grad_clipping(net, theta):
    """Clip the gradient"""
    params = [p for p in net.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad**2))for p in params))
    if norm > theta:
        for param in params:
            param.gard[:] *= theta / norm

def train_epoch(net, train_iter, loss, updater, device, use_random_iter):
    """Train a net within one epoch"""
    state, timer = None, Timer()
    metric = Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
           state = None
        else:
            if not isinstance(state[0], tuple):
                # `state` is a list of tensors for GRU
                for s in state:
                    s.detach_()
            else:
                # `state` is a list of tuples of tensors for ConvLSTM
                for h, c in state:
                    h.detach_()
                    c.detach_()
        X, y = X.to(device), Y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.backward()
        grad_clipping(net, 1)
        updater.step()
        metric.add(l * X.shape[0] * X.shape[1], X.shape[0] * X.shape[1])
    return metric[0] / metric[1], metric[1] / timer.stop()

def train(net, train_iter, lr, num_epochs, device, use_random_iter, path):
    """Train a model"""
    net.train()
    loss = nn.MSELoss()
    animator = Animator(xlabel='epoch', ylabel='MSE', legend=['train'], xlim=[10, num_epochs])

    updater = torch.optim.SGD(net.parameters(), lr)

    for epoch in range(num_epochs):
        MSE, speed = train_epoch(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, [MSE])
    print(f'MSE: {MSE:.6f}; {speed:.1f} pictures/sec on {str(device)}')

    with open(path + "\\log.txt","a") as f:
        f.write("\n-----------TRAIN_INFO-----------\n")
        f.write("learning_rate: \t\t"+str(lr)+"\n")
        f.write("num_epochs: \t\t"+str(num_epochs)+"\n")
        f.write("MSE: \t\t\t\t"+str(MSE)+"\n")
        f.write("speed: \t\t\t\t"+str(speed)+" pictures/sec \n")
        f.write("device: \t\t\t"+str(device)+"\n")

def evaluate(net, data_iter, device):
    net.eval()
    metric = Accumulator(2)

    with torch.no_grad():
        loss = nn.MSELoss()
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat, _ = net(X)
            l = loss(y, y_hat)
            metric.add(l * X.shape[0] * X.shape[1], X.shape[0] * X.shape[1])
    return metric[0] / metric[1]

def save(net, PATH):
    net = net.to('cpu')
    PATH += '\\Model.pt'
    torch.save(net.state_dict(), PATH)
    print('save successfully.')

def load(net, PATH, device):
    net.load_state_dict(torch.load(PATH, map_location=device))
    print('load successfully.')

def Get_PATH():
    PATH = 'Models\\' + str(time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time())))
    os.mkdir(PATH)
    return PATH
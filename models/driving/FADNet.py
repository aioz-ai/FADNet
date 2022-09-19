import math
from collections import OrderedDict
import torch.nn as nn
import torch
from utils.optim import get_optimizer, get_lr_scheduler
from ..model import Model
from torch.nn import functional as F
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
NUMBER_CLASSES = 1
FEATURE_SIZE = 6272
import tqdm
def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):

    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_needed = max(0, (out_rows - 1) * stride[0] + effective_filter_size_rows -
                  input_rows)
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)


class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

# Same padding 2D Convolutional (use this  class to define layer)
class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

# Normalize
cuda0 = torch.device('cuda:0')
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean, device=cuda0)
        self.std = torch.tensor(std, device=cuda0)

    def forward(self, input):
        x = input / 255.0
        x = x - self.mean
        x = x / self.std
        return x

class FADNet(nn.Module):
    def __init__(self):
        super(FADNet, self).__init__()
        self.conv1 = Conv2d(1, 32, (5, 5), stride=2)
        self.max_pool1 = nn.MaxPool2d((3, 3), 2)
        self.res_block1 = nn.Sequential(OrderedDict([
            ('batch_norm', nn.BatchNorm2d(32)),
            ('relu', nn.ReLU()),
            ('conv2d', Conv2d(32, 32, (3, 3), stride=2)),
            ('batch_norm_1', nn.BatchNorm2d(32)),
            ('relu_1', nn.ReLU()),
            ('conv2d_2', Conv2d(32, 32, (3, 3)))
        ]))
        self.conv2 = Conv2d(32, 256, (1, 1), stride=7)

        self.res_block2 = nn.Sequential(OrderedDict([
            ('batch_norm', nn.BatchNorm2d(32)),
            ('relu', nn.ReLU()),
            ('conv2d', Conv2d(32, 64, (3, 3), stride=2)),
            ('batch_norm_1', nn.BatchNorm2d(64)),
            ('relu_1', nn.ReLU()),
            ('conv2d_2', Conv2d(64, 64, (3, 3)))
        ]))
        self.conv3 = Conv2d(32, 256, (1, 1), stride=4)

        self.res_block3 = nn.Sequential(OrderedDict([
            ('batch_norm', nn.BatchNorm2d(64)),
            ('relu', nn.ReLU()),
            ('conv2d', Conv2d(64, 128, (3, 3), stride=2)),
            ('batch_norm_1', nn.BatchNorm2d(128)),
            ('relu_1', nn.ReLU()),
            ('conv2d_2', Conv2d(128, 128, (3, 3)))
        ]))
        self.conv4 = Conv2d(64, 256, (1, 1), stride=2)
        self.dropout = nn.Dropout2d(p=0.5)
        self.relu = nn.ReLU()

        self.fc_feature = nn.Linear(3, 1)
        self.fc = nn.Linear(FEATURE_SIZE, NUMBER_CLASSES)

    def forward(self, inputs):
        x1 = inputs
        x1 = self.conv1(x1)
        x1 = self.max_pool1(x1)

        x2 = self.res_block1(x1)

        f1 = self.conv2(x1)
        f1 = f1.view(inputs.shape[0], -1).reshape(inputs.shape[0], FEATURE_SIZE, -1)
        # GAP support feature 1
        f1 = f1.mean(axis=-1)

        x3 = self.res_block2(x2)

        f2 = self.conv3(x2)
        f2 = f2.view(inputs.shape[0], -1).reshape(inputs.shape[0], FEATURE_SIZE, -1)
        # GAP support feature 2
        f2 = f2.mean(axis=-1)

        x4 = self.res_block3(x3)

        f3 = self.conv4(x3)
        f3 = f3.view(inputs.shape[0], -1).reshape(inputs.shape[0], FEATURE_SIZE, -1)
        # GAP support feature 3
        f3 = f3.mean(axis=-1)

        x4 = self.relu(x4)
        x4 = self.dropout(x4)
        x4 = x4.view(inputs.shape[0], -1)

        # Support feature Accumulation
        x_feature = self.fc_feature(torch.stack([f1, f2, f3], axis=2)).squeeze(-1)

        # Aggregation with hadamard product
        x_final = torch.mul(x4, x_feature)
        # prediction
        return self.fc(x_final)

class DrivingNet(Model):
    def __init__(self, criterion, metric, device,
                 optimizer_name="adam", lr_scheduler="sqrt", initial_lr=1e-3, epoch_size=1):
        super(DrivingNet, self).__init__()
        self.net = FADNet().to(device)
        self.criterion = criterion
        self.metric = metric
        self.device = device

        self.optimizer = get_optimizer(optimizer_name, self.net, initial_lr)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, lr_scheduler, epoch_size)

    def fit_iterator_one_epoch(self, iterator):
        epoch_loss = 0
        epoch_acc = 0

        self.net.train()

        for x, y in iterator:
            self.optimizer.zero_grad()
            x = x.to(self.device)
            y = y.unsqueeze(-1).to(self.device)
            predictions = self.net(x)

            loss = self.criterion(predictions, y)

            acc = self.metric(predictions, y)

            loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def fit_batch(self, iterator, update=True):
        self.net.train()

        x, y = next(iter(iterator))

        x = x.to(self.device)
        y = y.unsqueeze(-1).to(self.device)

        self.optimizer.zero_grad()

        predictions = self.net(x)

        loss = self.criterion(predictions, y)

        acc = self.metric[0](predictions, y)

        loss.backward()

        if update:
            self.optimizer.step()
            # self.lr_scheduler.step()

        batch_loss = loss.item()
        batch_acc = acc.item()

        return batch_loss, batch_acc

    def evaluate_iterator(self, iterator):
        epoch_loss = 0
        epoch_acc = 0

        self.net.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm.tqdm(iterator)):
                x = x.to(self.device)
                y = y.unsqueeze(-1).to(self.device)
                predictions = self.net(x)

                loss = self.criterion(predictions, y)

                acc = self.metric[0](predictions, y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

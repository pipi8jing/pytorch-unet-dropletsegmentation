import torch
import torch.nn.functional as F
import cv2
import numpy


class BinaryDiceLoss(torch.nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(2*torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class Net(torch.nn.Module):
    """
        主网络架构
    """
    def __init__(self):
        super(Net, self).__init__()
        mult_channels = 32
        depth = 4
        self.net_main = Net_main(in_channels=1, mult_channels=mult_channels, depth=depth)
        self.conv_out = torch.nn.Conv2d(mult_channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x_main = self.net_main(x)
        x_conv = self.conv_out(x_main)
        return x_conv


class Net_main(torch.nn.Module):
    """
        网络实现
    """
    def __init__(self, in_channels, mult_channels=2, depth=0):
        """
        :param in_channels: 输入的通道数
        :param mult_channels: 决定输出通道数的系数
        :param depth: 深度，如果为0，这个子网将仅仅是使通道计数翻倍的卷积。
        """
        super(Net_main, self).__init__()
        self.depth = depth
        out_channels = in_channels*mult_channels
        self.sub_conv_horizon = Sub_conv_horizon(in_channels, out_channels)
        if self.depth > 0:
            self.last_conv = Sub_conv_horizon(2*out_channels, out_channels)
            self.sub_conv_vertical_down = Sub_conv_vertical_down(out_channels, out_channels)
            self.sub_conv_vertical_up = Sub_conv_vertical_up(2*out_channels, out_channels)
            self.sub_net = Net_main(out_channels, mult_channels=2, depth=(depth-1))

    def forward(self, x):
        if self.depth == 0:
            return self.sub_conv_horizon(x)
        else:
            x_horizon = self.sub_conv_horizon(x)
            x_vertical_down = self.sub_conv_vertical_down(x_horizon)
            x_sub_net = self.sub_net(x_vertical_down)
            x_vertical_up = self.sub_conv_vertical_up(x_sub_net)
            x_cat = torch.cat((x_horizon, x_vertical_up), 1)
            x_res = self.last_conv(x_cat)
            return x_res


class Sub_conv_horizon(torch.nn.Module):
    """
        每一层的两个卷积操作--水平方向
    """
    def __init__(self, n_in, n_out):
        super(Sub_conv_horizon, self).__init__()
        self.conv1 = torch.nn.Conv2d(n_in, n_out, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(n_out)
        self.relu1 = torch.nn.ReLU()
        #self.dropout1 = torch.nn.Dropout()
        self.conv2 = torch.nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(n_out)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout()



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        #x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return x


class Sub_conv_vertical_down(torch.nn.Module):
    """
        下采样网络
    """
    def __init__(self, n_in, n_out):
        super(Sub_conv_vertical_down, self).__init__()
        #self.pooling = torch.nn.MaxPool2d(2, 2)
        self.conv = torch.nn.Conv2d(n_in, n_out, kernel_size=2, stride=2)
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm2d(n_out)


    def forward(self, x):
        #x = self.pooling(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Sub_conv_vertical_up(torch.nn.Module):
    """
        上采样网络
    """
    def __init__(self, n_in, n_out):
        super(Sub_conv_vertical_up, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(n_in, n_out, kernel_size=2, stride=2)
        #self.conv2 = torch.nn.Conv2d(n_in, n_out, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(n_out)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        #x = F.interpolate(x, scale_factor=(2, 2), mode='bilinear')
        #x = self.conv2(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
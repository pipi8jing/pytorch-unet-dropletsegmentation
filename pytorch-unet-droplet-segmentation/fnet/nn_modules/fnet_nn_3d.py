# -*- coding:utf-8 -*-
# @Time : 2021/11/6 11:46
# @Author : Fang mirror
# @Site :
# @File : fnet_nn_3d.py
# @Software: PyCharm
import torch


class Net(torch.nn.Module):
    """
        主网络架构
    """
    def __init__(self):
        super(Net, self).__init__()
        mult_channels = 32
        depth = 4

        self.net_main = Net_main(in_channels=1, mult_channels=mult_channels, depth=depth)
        self.conv_out = torch.nn.Conv3d(mult_channels, 1, kernel_size=3, padding=1)
        self.conv_out = torch.nn.Conv3d(mult_channels, 1, kernel_size=3, padding=1)

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
        self.conv1 = torch.nn.Conv3d(n_in, n_out, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(n_out)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(n_out)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class Sub_conv_vertical_down(torch.nn.Module):
    """
        下采样网络
    """
    def __init__(self, n_in, n_out):
        super(Sub_conv_vertical_down, self).__init__()
        self.conv = torch.nn.Conv3d(n_in, n_out, kernel_size=2, stride=2)
        self.bn = torch.nn.BatchNorm3d(n_out)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
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
        self.conv = torch.nn.ConvTranspose3d(n_in, n_out, kernel_size=2, stride=2)
        self.bn = torch.nn.BatchNorm3d(n_out)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
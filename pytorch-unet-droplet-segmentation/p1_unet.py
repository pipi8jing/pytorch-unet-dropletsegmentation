import torch
import torch.nn.functional as F



class SoftDiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        #为了防止除0的发生
        smooth = 1

        #probs = F.sigmoid(logits)
        m1 = logits.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = score.sum() / num
        return score


class FocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = F.sigmoid(_input)
        pt = torch.clamp(pt, 1e-32, 0.999)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = 10 * torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class Net(torch.nn.Module):
    """
        主网络架构
    """
    def __init__(self):
        super(Net, self).__init__()
        mult_channels = 16
        depth = 4
        self.net_main = Net_main(in_channels=2*mult_channels, mult_channels=1, depth=depth)
        self.conv_out = torch.nn.Conv2d(2*mult_channels, 1, kernel_size=1)
        #self.conv_in = torch.nn.Conv2d(2, 1, kernel_size=1)
        self.sub_conv_vertical_up = Sub_conv_vertical_up(1, mult_channels)
        self.sub_conv_horizon1 = Sub_conv_horizon(mult_channels, mult_channels)
        self.sub_conv_horizon2 = Sub_conv_horizon(1, mult_channels)
        #self.sub_conv_horizon3 = Sub_conv_horizon(2*mult_channels, mult_channels)
        self.sub_conv_vertical_down = Sub_conv_vertical_down(mult_channels, mult_channels)

    def forward(self, x):
        #x = x.squeeze()
        #x = self.conv_in(x)
        x_over = self.sub_conv_vertical_up(x)
        x_over = self.sub_conv_horizon1(x_over)
        x_over = self.sub_conv_vertical_down(x_over)
        x_conv = self.sub_conv_horizon2(x)
        x_cat = torch.cat((x_over, x_conv), 1)
        x_main = self.net_main(x_cat)
        x_final = self.conv_out(x_main)
        return x_final


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
        self.dropout2 = torch.nn.Dropout(p=0.2)



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
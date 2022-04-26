import importlib
import os
import torch


torch.set_printoptions(profile="full")


class Model(object):
    def __init__(self, nn_module=None, init_weights=False, lr=0.001,
                 criterion_fn=torch.nn.MSELoss, nn_kwargs={}, gpu_ids=-1):
        self.nn_module = nn_module
        self.init_weights = init_weights
        self.lr = lr
        self.criterion = criterion_fn()
        self.gpu_ids = [gpu_ids] if isinstance(gpu_ids, int) else gpu_ids
        self.nn_kwargs = nn_kwargs

        self.device = torch.device('cuda', self.gpu_ids[0]) if self.gpu_ids[0] >= 0 else torch.device('cpu')
        self.count_iter = 0
        # 模型初始化主函数
        self._init_model(nn_kwargs=self.nn_kwargs)

    def __str__(self):
        out_str = f'{self.nn_module} | {str(self.nn_kwargs)} | {self.count_iter}'
        return out_str

    def _init_model(self, nn_kwargs={}):
        """
        根据模型名称直接链接到nn_modules中加载模型，并初始化权重
        在此利用Adam优化器进行优化
        :param nn_kwargs: 模型传输参数
        :return:
        """
        if self.nn_module is None:
            self.net = None
            return
        self.net = importlib.import_module('fnet.nn_modules.' + self.nn_module).Net(**nn_kwargs)
        if self.init_weights:
            self.net.apply(_weights_init)
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.5, 0.999))

    def train_iter(self, signal, target):
        """
        模型训练主函数
        :param signal: 明场数据
        :param target: 荧光数据
        :return: 返回Loss
        """
        signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
        target = torch.tensor(target, dtype=torch.float32, device=self.device)
        if len(self.gpu_ids) > 1:
            module = torch.nn.DataParallel(self.net, device_ids=self.gpu_ids)
        else:
            module = self.net
        module.train()
        self.optimizer.zero_grad()
        output = module(signal)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        self.count_iter += 1
        return loss.item()

    def predict(self, signal):
        """
        模型预测主函数
        :param signal:明场数据
        :return: 荧光数据
        """
        signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
        if len(self.gpu_ids) > 1:
            module = torch.nn.DataParallel(self.net, device_ids=self.gpu_ids)
        else:
            module = self.net
        module.eval()
        with torch.no_grad():
            prediction = module(signal).cpu()
        return prediction

    def load_state(self, path_load, gpu_ids=-1):
        """
        加载预训练模型状态函数
        :param path_load: 加载路径
        :param gpu_ids:
        :return:
        """
        state_dict = torch.load(path_load)
        self.nn_module = state_dict['nn_module']
        self.nn_kwargs = state_dict.get('nn_kwargs', {})
        self._init_model(nn_kwargs=self.nn_kwargs)
        self.net.load_state_dict(state_dict['nn_state'])
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        self.count_iter = state_dict['count_iter']
        self.to_gpu(gpu_ids)

    def to_gpu(self, gpu_ids):
        if isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]
        self.gpu_ids = gpu_ids
        self.device = torch.device('cuda', self.gpu_ids[0]) if self.gpu_ids[0] >= 0 else torch.device('cpu')
        self.net.to(self.device)
        _set_gpu_recursive(self.optimizer.state, self.gpu_ids[0])  # this may not work in the future

    def save_state(self, path_save):
        """
        保存模型状态
        :param path_save: 保存路径
        :return:
        """
        curr_gpu_ids = self.gpu_ids
        dirname = os.path.dirname(path_save)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.to_gpu(-1)
        torch.save(self.get_state(), path_save)
        self.to_gpu(curr_gpu_ids)

    def get_state(self):
        return dict(
            nn_module=self.nn_module,
            nn_kwargs=self.nn_kwargs,
            nn_state=self.net.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            count_iter=self.count_iter,
        )


def _weights_init(m):
    """
    权重初始化方法
    :param m:
    :return:
    """
    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def _set_gpu_recursive(var, gpu_id):
    """Moves Tensors nested in dict var to gpu_id.

    Modified from pytorch_integrated_cell.

    Parameters:
    var - (dict) keys are either Tensors or dicts
    gpu_id - (int) GPU onto which to move the Tensors
    """
    for key in var:
        if isinstance(var[key], dict):
            _set_gpu_recursive(var[key], gpu_id)
        elif torch.is_tensor(var[key]):
            if gpu_id == -1:
                var[key] = var[key].cpu()
            else:
                var[key] = var[key].cuda(gpu_id)

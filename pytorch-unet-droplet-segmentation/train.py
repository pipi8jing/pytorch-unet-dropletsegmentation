import argparse
import json
import os
import numpy as np
import torch
import fnet
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def get_dataloader(remaining_iterations, opts, validation=False):
    # 数据集初始化
    transform_signal = [eval(t) for t in opts.transform_signal]
    transform_target = [eval(t) for t in opts.transform_target]
    ds = getattr(fnet.data, opts.class_dataset)(
        img_path=opts.path_dataset,  # 训练数据地址
        flag=True if not validation else False,  # 是否存在验证集
        transform_signal=transform_signal,
        transform_target=transform_target
    )
    print(ds)
    # 读取的图像数据较大，需要进行裁剪，采用随机裁剪的方式
    ds_patch = fnet.data.BufferedPatchDataset(
        dataset=ds,
        patch_size=opts.patch_size,
        buffer_size=opts.buffer_size if not validation else len(ds),
        buffer_switch_frequency=opts.buffer_switch_frequency if not validation else -1,
        npatches=remaining_iterations * opts.batch_size if not validation else 4 * opts.batch_size,
        verbose=False,
        shuffle_images=opts.shuffle_images,
        **opts.bpds_kwargs
    )
    # 加载数据集
    dataloader = torch.utils.data.DataLoader(
        ds_patch,
        batch_size=opts.batch_size,
    )
    return dataloader


def main():
    # 读取配置文件
    with open("./config/train.json", "r") as json_file:
        train_options = json.load(json_file)
    opts = argparse.Namespace()
    opts.__dict__.update(train_options)

    # tensorboard调试展示
    writer = SummaryWriter("logs")

    # 计时功能
    time_start = datetime.now()

    # 模型保存文件夹路径
    if not os.path.exists(opts.path_run_dir):
        os.makedirs(opts.path_run_dir)
    # 是否需要中间模型
    if len(opts.iter_checkpoint) > 0:
        path_checkpoint_dir = os.path.join(opts.path_run_dir, 'chechpoints')
        if not os.path.exists(path_checkpoint_dir):
            os.makedirs(path_checkpoint_dir)

    # 设定随机种子
    if opts.seed is not None:
        np.random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed_all(opts.seed)

    # 实例化模型
    path_model = os.path.join(opts.path_run_dir, 'model.p')
    if os.path.exists(path_model):
        model = fnet.load_model_from_dir(opts.path_run_dir, gpu_ids=opts.gpu_ids)
    else:
        model = fnet.fnet_model.Model(
            nn_module=opts.nn_module,
            lr=opts.lr,
            gpu_ids=opts.gpu_ids,
            nn_kwargs=opts.nn_kwargs
        )

    # 保存loss
    path_loss_csv = os.path.join(opts.path_run_dir, 'loss.csv')
    if os.path.exists(path_loss_csv):
        fnetlogger = fnet.FnetLogger(path_loss_csv)
    else:
        fnetlogger = fnet.FnetLogger(columns=['num_iter', 'loss_batch'])

    # 剩余迭代次数
    n_remaining_iterations = max(0, (opts.n_iter - model.count_iter))
    # 加载训练集
    dataloader_train = get_dataloader(n_remaining_iterations, opts)
    # 是否存在验证集
    if opts.path_dataset_val:
        dataloader_val = get_dataloader(n_remaining_iterations, opts, validation=True)
        criterion_val = model.criterion
        path_loss_val_csv = os.path.join(opts.path_run_dir, 'loss_val.csv')
        if os.path.exists(path_loss_val_csv):
            fnetlogger_val = fnet.FnetLogger(path_loss_val_csv)
        else:
            # 在这里可以保存各项精度信息栏
            fnetlogger_val = fnet.FnetLogger(columns=['num_iter', 'loss_val'])
    # 保存参数信息
    with open(os.path.join(opts.path_run_dir, 'train_options.json'), 'w') as info:
        json.dump(vars(opts), info, indent=4, sort_keys=True)

    # 开始执行训练迭代过程
    minloss = 1
    bestiter = 0
    for i, (signal, target) in enumerate(dataloader_train, model.count_iter):
        loss_batch = model.train_iter(signal, target)
        fnetlogger.add({'num_iter': i + 1, 'loss_batch': loss_batch})
        # 利用tensorboard记录loss
        writer.add_scalar("train_loss", loss_batch, i + 1)
        # 每隔定量数据显示一次Loss
        if i % 100 == 0:
            print('num_iter: {:6d} | loss_batch: {:.3f}'.format(i + 1, loss_batch))
        # 每隔一定时间点保存状态
        if ((i + 1) % opts.interval_save == 0) or ((i + 1) == opts.n_iter):
            model.save_state(path_model)
            fnetlogger.to_csv(path_loss_csv)
            if opts.path_dataset_val:
                loss_val_sum = 0
                for idx_val, (signal_val, target_val) in enumerate(dataloader_val):
                    pred_val = model.predict(signal_val)
                    loss_val_batch = criterion_val(pred_val, target_val).item()
                    loss_val_sum += loss_val_batch
                    print('loss_val_batch: {:.3f}'.format(loss_val_batch))
                loss_val = loss_val_sum / len(dataloader_val)
                if i+1 > 1000 and loss_val < minloss:
                    minloss = loss_val
                    bestiter = i + 1
                    path_save_checkpoint = os.path.join(path_checkpoint_dir, 'model_max{:06d}.p'.format(i + 1))
                    model.save_state(path_save_checkpoint)
                print('loss_val: {:.3f}'.format(loss_val))
                writer.add_scalar("val_loss", loss_val, i + 1)
                fnetlogger_val.add({'num_iter': i + 1, 'loss_val': loss_val})
                fnetlogger_val.to_csv(path_loss_val_csv)
                for tag, value in model.net.named_parameters():
                    tag = tag.replace('.', '/')
                    #writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), i + 1)
                    #writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), i + 1)
                writer.add_scalar('learning_rate', model.optimizer.param_groups[0]['lr'], i + 1)
        # 保存中间断点
        if (i + 1) in opts.iter_checkpoint:
            path_save_checkpoint = os.path.join(path_checkpoint_dir, 'model_{:06d}.p'.format(i + 1))
            model.save_state(path_save_checkpoint)
    writer.close()
    print(bestiter)
    print(minloss)
    print(datetime.now())


if __name__ == '__main__':
    main()

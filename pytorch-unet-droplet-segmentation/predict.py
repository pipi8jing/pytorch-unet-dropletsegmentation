import argparse
import json
import os
import cv2
import tifffile
import torch
import fnet.transforms


def get_dataset(opts, propper):
    transform_signal = [eval(t) for t in opts.transform_signal]
    transform_target = [eval(t) for t in opts.transform_target]
    transform_signal.append(propper)
    transform_target.append(propper)
    ds = getattr(fnet.data, opts.class_dataset)(
        img_path=opts.path_dataset,
        transform_signal=transform_signal
    )
    print(ds)
    return ds


def save_tiff_and_log(tag, ar, path_tiff_dir, path_log_dir):
    if not os.path.exists(path_tiff_dir):
        os.makedirs(path_tiff_dir)
    path_tiff = os.path.join(path_tiff_dir, '{:s}.tiff'.format(tag))
    tifffile.imsave(path_tiff, ar)
    print('saved:', path_tiff)


def save_tiff(tag, ar, path_tiff_dir):
    ar = ar[0]
    path_tiff = os.path.join(path_tiff_dir, '{:s}.tiff'.format(tag))
    maxValue = 1
    threshValue = 0.5
    ret1, ar = cv2.threshold(ar, threshValue, maxValue, cv2.THRESH_BINARY)
    tifffile.imsave(path_tiff, ar)
    print('saved:', path_tiff)


def main():
    # 读取配置文件
    with open("./config/predict.json", "r") as json_file:
        train_options = json.load(json_file)
    opts = argparse.Namespace()
    opts.__dict__.update(train_options)

    # 如果保存路径存在则返回
    if os.path.exists(opts.path_save_dir):
        print('Output path already exists.')
        return
    # 对图像进行填充和裁剪
    if opts.class_dataset == 'TiffDatasetTest':
        if opts.propper_kwargs.get('action') == '-':
            opts.propper_kwargs['n_max_pixels'] = 67108864
    propper = fnet.transforms.Propper(**opts.propper_kwargs)
    print(propper)
    model = None
    # 加载数据集，在图像变换中加上填充或裁剪操作
    dataset = get_dataset(opts, propper)
    # 数据集长度
    indices = range(len(dataset)) if opts.n_images < 0 else range(min(opts.n_images, len(dataset)))
    for idx in indices:
        data = [torch.unsqueeze(d, 0) for d in dataset[idx]]  # make batch of size 1
        # 明场数据对应第0位
        signal = data[0]
        target = data[1] if (len(data) > 1) else None
        # 文件保存路径
        path_tiff_dir = os.path.join(opts.path_save_dir, '{:02d}'.format(idx))
        # 保存原始数据到指定路径
        if not opts.no_signal:
            save_tiff_and_log('signal', signal.numpy()[0,], path_tiff_dir, opts.path_save_dir)
        if not opts.no_target and target is not None:
            save_tiff_and_log('target', target.numpy()[0,], path_tiff_dir, opts.path_save_dir)

        # 模型所在路径
        for path_model_dir in opts.path_model_dir:
            if (path_model_dir is not None) and (model is None or len(opts.path_model_dir) > 1):
                # 加载模型
                model = fnet.load_model(path_model_dir, opts.gpu_ids, module=opts.module_fnet_model)
                print(model)
                name_model = os.path.basename(path_model_dir)
            # 预测结果输出为prediction
            prediction = model.predict(signal) if model is not None else None
            if not opts.no_prediction and prediction is not None:
                # 保存预测结果
                save_tiff_and_log('prediction_{:s}'.format(name_model), prediction.numpy()[0], path_tiff_dir,
                                  opts.path_save_dir)
                # 若需要二值化则打开此步操作
                save_tiff('prediction_bw{:s}'.format(name_model), prediction.numpy()[0], path_tiff_dir)
            if not opts.no_prediction_unpropped:
                ar_pred_unpropped = propper.undo_last(prediction.numpy()[0, 0,])
                save_tiff_and_log('prediction_{:s}_unpropped'.format(name_model), ar_pred_unpropped, path_tiff_dir
                                  , opts.path_save_dir)


if __name__ == '__main__':
    main()
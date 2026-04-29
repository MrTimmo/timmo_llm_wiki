import os
import sys
import argparse
from mindspore import context
# 数据相关
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype
import json
import numpy as np
# 模型相关：模型结构、损失函数、优化器等
import mindspore.nn as nn
# 模型结构(自定义)
from mlp_net import MLPNet
# 训练的checkpoint
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
# 训练、评估
from mindspore.train.callback import LossMonitor
from mindspore import Model
from mindspore.nn import Accuracy

    

# 基础环境配置
parser = argparse.ArgumentParser(description='MindSpore MLPNet Example')
parser.add_argument('--device_target', type=str, default='CPU', choices=['Ascend', 'GPU', 'CPU'])
args = parser.parse_known_args()[0]
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
DATA_PATH = "datasets/MNIST_Data"

def print_MNIST(mnist_ds):
    for data in mnist_ds.create_dict_iterator(output_numpy=True):
        output = {
            "image": {
                "shape": list(data['image'].shape),
                "dtype": str(data['image'].dtype),
                "data": data['image'].squeeze().tolist()  # 去掉单通道维度
            },
            "label": int(data['label'])
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        break

# 数据处理过程
def create_dataset(data_path, batch_size=64, repeat_size=1,
                   num_parallel_workers=1):
    # 获取数据集
    mnist_ds = ds.MnistDataset(data_path)

    print(mnist_ds.output_shapes())
    print(mnist_ds.get_dataset_size())
    # 创建字典迭代器,取第一条
    #print_MNIST(mnist_ds)

    # 定义resize的关键参数
    resize_height, resize_width = 32, 32
    ## 定义rescale的关键参数
    rescale = 1.0 / 255.0
    # 0.1307​ 是 MNIST 训练集归一化后的像素均值
    # 0.3081​ 是 MNIST 训练集归一化后的像素标准差
    # 通过此变换，输入数据会被转换为均值为 0、标准差为 1​ 的标准正态分布
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # 定义resize算子，使用双线性插值
    resize_op = vision.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    # 定义rescale算子，进行均值为0，标准差为1的归一化，有助于模型收敛
    rescale_nml_op = vision.Rescale(rescale_nml * rescale, shift_nml)
    # 定义hwc变成chw数据格式的变换
    hwc2chw_op = vision.HWC2CHW()
    type_cast_op = transforms.TypeCast(mstype.int32)
    # 将label列转换成int32类型
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    # 对image的输入做三次操作，分别是[28,28,1]转换成[32,32,1],然后对像素数据记性标准归一化， 最后将hwc 转换成chw
    mnist_ds = mnist_ds.map(operations=[resize_op, rescale_nml_op, hwc2chw_op], input_columns="image",num_parallel_workers=num_parallel_workers)

    #print_MNIST(mnist_ds)

    

    # 进行shuffle、batch操作
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=False)
    print(mnist_ds.get_dataset_size())
    return mnist_ds

def train_net(args, model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
    # 创建数据集
    dataset = create_dataset(data_path=data_path, repeat_size=repeat_size)
    # 训练模型
    model.train(epoch_size, dataset, callbacks=[ckpoint_cb, LossMonitor(125)], dataset_sink_mode=sink_mode)

def test_net(model, data_path):
    """定义验证的方法"""
    ds_eval = create_dataset(os.path.join(data_path, "test"))
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("{}".format(acc))


if __name__ == "__main__":
    # 1. 训练过程
    train_epoch = 10
    dataset_size = 1
    # 创建模型结构
    net = MLPNet()
    # 定义损失函数
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    # 定义优化器
    net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
    # 定义模型
    model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    # 设置模型保存参数
    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
    # 应用模型保存参数
    ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)
    train_net(args, model, train_epoch, "datasets/MNIST_Data/train", dataset_size, ckpoint, False)

    test_net(model,  "datasets/MNIST_Data")



    
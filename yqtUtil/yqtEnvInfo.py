# 模板
# 打印环境信息

import torch
import sys


def printInfo():
    print("python 版本\t", sys.version)
    print("python 解释器位置\t", sys.executable)
    print("torch 版本\t", torch.__version__)
    print("cuda 版本\t", torch.version.cuda)
    print("是否有gpu\t", torch.cuda.is_available())
    print("gpu可用数\t", torch.cuda.device_count())
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        GPU_device = torch.cuda.get_device_properties(device)
        print("gpu 型号\t\t", GPU_device.name)
        print("gpu 内存\t\t", GPU_device.total_memory / (1024 ** 2), "MB")


def yqtDevice():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    return device


if __name__ == '__main__':
    printInfo()

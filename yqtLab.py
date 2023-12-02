import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import shutil

import yqtUtil.yqtDataset as yqtDataset
import yqtUtil.yqtEnvInfo as yqtEnvInfo
import yqtUtil.yqtNet as yqtNet
import yqtUtil.yqtRun as yqtRun

import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
import sys, os

CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../../src")

# 一些外部参数的设置
# batchsize
train_batchsize = 100
test_batchsize = 100
# 打印间隔
train_print_freq = 1
test_print_freq = 1

# 清除目录，每次重新保存模型和log
if os.path.exists("best.pth"):
    os.remove("best.pth")
if os.path.exists("logs_model"):
    shutil.rmtree("logs_model")
os.mkdir("logs_model")

yqtEnvInfo.printInfo()
device = yqtEnvInfo.yqtDevice()

# 加载数据
netParams = snn.params('network.yaml')
train_dataset = yqtDataset.yqtDataset(datasetPath=netParams['training']['path']['in'],
                                      sampleFile=netParams['training']['path']['train'],
                                      samplingTime=netParams['simulation']['Ts'],
                                      sampleLength=netParams['simulation']['tSample'])

test_dataset = yqtDataset.yqtDataset(datasetPath=netParams['training']['path']['in'],
                                     sampleFile=netParams['training']['path']['test'],
                                     samplingTime=netParams['simulation']['Ts'],
                                     sampleLength=netParams['simulation']['tSample'])

train_loader = DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=test_batchsize, shuffle=False, num_workers=0)

print("train_dataset size:\t", train_dataset.size)
print("test_dataset size:\t", test_dataset.size)

# 网络初始化
model = yqtNet.yqtNet(netParams).to(device)

if os.path.exists("best.pth"):
    model.load_state_dict(torch.load("best.pth"))
model = model.to(device)

# 训练和测试 保存权重
optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
criterion = snn.loss(netParams).to(device)

writer = SummaryWriter("logs_model")

# 分类任务 loss accuracy
best_prec = 0
if os.path.exists("best.pth"):
    best_prec = yqtRun.test(model=model, dataLoader=test_loader, criterion=criterion, print_freq=test_print_freq,
                            epoch=0, device=device, writer=writer)

for epoch in range(0, 20):
    yqtRun.train(model=model, dataLoader=train_loader, criterion=criterion, optimizer=optimizer, epoch=epoch,
                 print_freq=train_print_freq, device=device, writer=writer)
    prec_ = yqtRun.test(model=model, dataLoader=test_loader, criterion=criterion, print_freq=test_print_freq,
                        epoch=epoch, device=device, writer=writer)

    if prec_ > best_prec:
        best_prec = prec_
        torch.save(model.state_dict(), "best.pth")

writer.close()
print('train end Best accuracy: ', best_prec)

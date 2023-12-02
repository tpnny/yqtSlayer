import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yqtUtil.yqtDataset as yqtDataset
import yqtUtil.yqtEnvInfo as yqtEnvInfo
import yqtUtil.yqtNet as yqtNet
import slayerSNN as snn

test_batchsize = 1

yqtEnvInfo.printInfo()
device = yqtEnvInfo.yqtDevice()

# 建立测试数据集

# 加载数据
netParams = snn.params('network.yaml')
test_dataset = yqtDataset.yqtDataset(datasetPath=netParams['training']['path']['in'],
                                     sampleFile=netParams['training']['path']['test'],
                                     samplingTime=netParams['simulation']['Ts'],
                                     sampleLength=netParams['simulation']['tSample'])
test_loader = DataLoader(test_dataset, batch_size=test_batchsize, shuffle=False, num_workers=0)

# 网络初始化

model = yqtNet.yqtNet(netParams).to(device)
model.load_state_dict(torch.load("best.pth"))
model = model.to(device)

# 模型保存成onnx
# input_ = torch.randn(1, 2312, 1, 1, 300)
# torch.onnx.export(model,  # 搭建的网络
#                   input_,  # 输入张量
#                   'model_.onnx',  # 输出模型名称
#                   input_names=["input"],  # 输入命名
#                   output_names=["output"],  # 输出命名
#                   dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})

# 打印模型参数
x_ = sum([param.nelement() for param in model.parameters()]) / 1e6
print("Number of parameter: %.2fM" % x_)

writer = SummaryWriter("logs_model")
out_ = []
label_ = []
model.eval()
with torch.no_grad():
    for i, (input, target, label) in enumerate(test_loader):
        target_ = target.to(device)
        input_ = input.to(device)
        output = model(input_)

        o_ = snn.predict.getClass(output).cpu().item()
        t_ = label.cpu().item()

        writer.add_scalars("output", {'pred': o_}, i)
        writer.add_scalars("output", {'label': t_}, i)
        out_.append(o_)
        label_.append(t_)

out_ = np.array(out_)
label_ = np.array(label_)
merge_ = np.vstack((out_, label_)).T
np.savetxt("pred_label.txt", merge_)
print(merge_)
writer.close()

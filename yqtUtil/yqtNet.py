# 模板
# 定义网络结构

import torch
import torch.nn as nn
import slayerSNN as snn


class yqtNet(nn.Module):
    def __init__(self, netParams):
        super(yqtNet, self).__init__()
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        self.fc1 = slayer.dense((34 * 34 * 2), 512)
        self.fc2 = slayer.dense(512, 10)

    def forward(self, spikeInput):
        spikeLayer1 = self.slayer.spike(self.slayer.psp(self.fc1(spikeInput)))
        spikeLayer2 = self.slayer.spike(self.slayer.psp(self.fc2(spikeLayer1)))
        return spikeLayer2


if __name__ == '__main__':
    netParams = snn.params('/data/Codes/sync/20230305_yqtSlayer/network.yaml')

    net = yqtNet(netParams).to("cuda")
    input_ = torch.randn([1, 2312, 1, 1, 300]).to("cuda")

    out = net(input_)
    print(out)
    print(out.shape)

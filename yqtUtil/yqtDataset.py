# 模板
# 自定义数据集
# 可以比较灵活，这里仅做一个示例，实际设计的过程要主动地分离代码和数据
# 然后数据最好是单个文件，这样相当于压缩，读写更快
# 一般最后增减一下维度匹配上网络就可以

from torch.utils.data import Dataset
import numpy as np
import torch
import slayerSNN as snn


# 默认的 nmnist 数据转换方式
# 事件转成2维的 0 1 标志数据
# 再转成事件序列的数据

class yqtDataset(Dataset):
    def __init__(self, datasetPath, sampleFile, samplingTime, sampleLength):
        self.path = datasetPath
        self.samples = np.loadtxt(sampleFile).astype('int')
        self.samplingTime = samplingTime
        self.nTimeBins = int(sampleLength / samplingTime)
        self.size = self.samples.shape[0]

    def __getitem__(self, index):
        inputIndex = self.samples[index, 0]
        classLabel = self.samples[index, 1]

        inputSpikes = snn.io.read2Dspikes(self.path + str(inputIndex.item()) + '.bs2').toSpikeTensor(
            torch.zeros((2, 34, 34, self.nTimeBins)), samplingTime=self.samplingTime)
        desiredClass = torch.zeros((10, 1, 1, 1))
        desiredClass[classLabel, ...] = 1
        return inputSpikes.reshape((-1, 1, 1, inputSpikes.shape[-1])), desiredClass, classLabel

    def __len__(self):
        return self.samples.shape[0]


if __name__ == '__main__':
    dataPath = "/data/Data/NMNISTsmall/"
    trainSampleFile = "/data/Data/NMNISTsmall/train1K.txt"
    testSampleFile = "/data/Data/NMNISTsmall/test100.txt"
    samplingTime = 1.0
    sampleLength = 300

    aa = yqtDataset(datasetPath=dataPath, sampleFile=trainSampleFile, samplingTime=samplingTime,
                    sampleLength=sampleLength)
    print(aa.size)
    print(aa[0][0].shape)

    aa = yqtDataset(datasetPath=dataPath, sampleFile=testSampleFile, samplingTime=samplingTime,
                    sampleLength=sampleLength)
    print(aa.size)
    print(aa[1][0].shape)

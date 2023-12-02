# 模板
# 定义训练测试的过程
# 需要传入需要调整的参数
# 训练一次的模板，可以灵活调整，主要是分类和回归两种
# 训练分为两个类型，一个是分类，一个是回归，实际过程也是会机动

# 定义一次测试的过程
# 需要传入需要调整的参数
# 训练一次的模板，可以灵活调整，主要是分类和回归两种
# 训练分为两个类型，一个是分类，一个是回归，实际过程也是会机动
# 返回整个测试集的准确率 loss等信息

import torch
import time
import slayerSNN as snn


def train(model, dataLoader, criterion, optimizer, epoch, print_freq, writer, device="cpu"):
    batch_time = yqtAverage("time")
    losses = yqtAverage("loss")
    top1 = yqtAverage("top1")

    model.train()

    for i, (input, target, label) in enumerate(dataLoader):

        end = time.time()
        target_ = target.to(device)
        input_ = input.to(device)

        output = model(input_)
        # loss = criterion(output, target_)

        loss = criterion.numSpikes(output, target)

        # 调用准确率函数，直接计算top1的准确率，如果要获取对应的argmax，可以修改准确率函数，额外返回索引张量
        # 判断判对的总数

        correctSamples = torch.sum(snn.predict.getClass(output) == label).data.item()
        eq_ = correctSamples / target_.size(0)

        # 更新记录，传入的数据是当前batch的平均值
        top1.update(eq_, target_.size(0))
        losses.update(loss.data.item(), target_.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 这里的时间是整个bacth的时间，所以对应的计算的时候，不用再加batchsize,对应的输出信息也是一整个bacth的计算时间
        batch_time.update(time.time() - end)

        if (i + 1) % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.4f} ({top1.avg:.4f})'.format(epoch, i + 1, len(dataLoader), batch_time=batch_time,
                                                                  loss=losses, top1=top1))

    # 当前epoch计算完的信息
    print('Epoch: [{0}]\t'
          'Loss {loss.avg:.4f}\t'
          'Prec@1  {top1.avg:.4f}'.format(epoch, loss=losses, top1=top1))

    writer.add_scalars("loss", {'train': losses.avg}, epoch)
    writer.add_scalars("prec", {'train': top1.avg}, epoch)


def test(model, dataLoader, criterion, print_freq, writer, epoch, device="cpu"):
    batch_time = yqtAverage()
    losses = yqtAverage()
    top1 = yqtAverage()

    model.eval()
    with torch.no_grad():
        for i, (input, target, label) in enumerate(dataLoader):
            end = time.time()
            target_ = target.to(device)
            input_ = input.to(device)

            output = model(input_)
            # loss = criterion(output, target_)
            loss = criterion.numSpikes(output, target)

            correctSamples = torch.sum(snn.predict.getClass(output) == label).data.item()

            eq_ = correctSamples / target_.size(0)
            top1.update(eq_, target_.size(0))

            losses.update(loss.data.item(), target_.size(0))
            batch_time.update(time.time() - end)

            if i + 1 % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.4f} ({top1.avg:.4f})'.format(i + 1, len(dataLoader), batch_time=batch_time,
                                                                      loss=losses, top1=top1))

    print('Test prec {top1.avg:.3f},'.format(top1=top1))
    print('Test loss {Loss.avg:.3f}'.format(Loss=losses))
    writer.add_scalars("loss", {'test': losses.avg}, epoch)
    writer.add_scalars("prec", {'test': top1.avg}, epoch)
    return top1.avg


class yqtAverage(object):

    def __init__(self, name="loss"):
        self.name = name
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    # 可以用于统计耗时、损失、精度等平均值，训练或者测试时，每个batch积累到一个完整的epoch
    # loss默认的参数返回的是整个batch的平均值，所以累计的时候，需要显式指定实际的样本数量
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 分类任务时的单个batch正确率计算
# 该函数目标是计算前k个结果中是否命中了目标类别

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k / batch_size)
    return res


if __name__ == "__main__":
    losses = yqtAverage("loss")
    loss_list = [0.5, 0.4, 0.5, 0.6, 1]
    batch_size = 2
    for los in loss_list:
        losses.update(los, batch_size)
        print(losses.sum)
        print(losses.avg)

    output = torch.tensor([[10.5816, -0.3873, -1.0215, -1.0145, 0.4053], [0.7265, 0.4164, 1.3443, 1.2035, 0.8823],
                           [-0.4451, 0.1673, 1.2590, -2.0757, 1.7255], [0.2021, 0.3041, 0.1383, 0.3849, -1.6311]])
    target = torch.tensor([[4], [4], [2], [1]])

    print(accuracy(output, target, topk=(1, 2,)))

---
title: GPU in Pytorch  并行和分布式实践
toc: true
mathjax: true
top: true
cover: true
date: 2020-11-25 15:07:07
updated:
categories: Pytorch
tags:
	- Pytorch
	- CUDA
	- GPU
---

　　虽然还没有机会用到CUDA集群，但是前段时间对协程和并行化的研究，让我忍不住想要探索一下如何在多个GPU下利用Pytorch加快训练的实践方法，算是为之后并行训练优化一个理论参考吧！

　　Pytorch 大体上有3种实现并行的接口（另外还有一种不利用接口的拼接模型的技巧，之后再单独讨论），分别是：torch.multiprocessing, nn.DataParallel, nn.parallel.DistributedDataParallel，如果是是GPU多卡运行，最佳实践是 nn.parallle.DistributedDataParallel,官方文档 [CUDA SEMANTIC](https://pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead) 是这么描述的：

> **Use nn.parallel.DistributedDataParallel instead of multiprocessing or nn.DataParallel**
>
> Most use cases involving batched inputs and multiple GPUs should default to using [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) to utilize more than one GPU.
>
> There are significant caveats to using CUDA models with [`multiprocessing`](https://pytorch.org/docs/stable/multiprocessing.html#module-torch.multiprocessing); unless care is taken to meet the data handling requirements exactly, it is likely that your program will have incorrect or undefined behavior.
>
> It is recommended to use [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel), instead of [`DataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel) to do multi-GPU training, even if there is only a single node.
>
> The difference between [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) and [`DataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel) is: [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) uses multiprocessing where a process is created for each GPU, while [`DataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel) uses multithreading. By using multiprocessing, each GPU has its dedicated process, this avoids the performance overhead caused by GIL of Python interpreter.
>
> If you use [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel), you could use torch.distributed.launch utility to launch your program, see [Third-party backends](https://pytorch.org/docs/stable/distributed.html#distributed-launch).



### torch.multiprocessing

　　torch.multiprocessing 是 Python下 multiprocessing 的替代品，接口基本一致，并根据情况进行扩张，建议使用 python: multiprocessing.Queue 在进程中传递Pytorch对象。除此之外，还有很多坑，相见官方笔记 [MULTIPROCESSING BEST PRACTICES](https://pytorch.org/docs/stable/notes/multiprocessing.html), [并行处理最佳实践](https://pytorch.apachecn.org/docs/1.4/64.html)

```python
import torch.multiprocessing as mp
from model import MyModel

def train(model):
    # Construct data_loader, optimizer, etc.
    for data, labels in data_loader:
        optimizer.zero_grad()
        loss_fn(model(data), labels).backward()
        optimizer.step()  # This will update the shared parameters

if __name__ == '__main__':
    num_processes = 4
    model = MyModel()
    # NOTE: this is required for the ``fork`` method to work
    model.share_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(model,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
```



### DataParallel

　　DataParallel原理是，把model 副本copy到所有GPU上，其中每个GPU消耗数据的不同分区，相当于SIMD，把数据条目根据GPU数量重新分配

```python
model = nn.DataParallel(model)
```

　　代码验证 outside model 数据维度  和 inside model 维度

```python
class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output
        

model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)

for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
    
# 2 GPUs
# on 2 GPUs
Let's use 2 GPUs!
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
    In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

优点是，通过消耗输入数据不同分区的方式加快训练过程，缺点是，如果单个模型太大，无法放入单个GPU，就无法运行，这时有个Trick可以把模型分段载入不同GPU，利用拼接的方式完成并行训练，该模型将单个模型拆分到不同的 GPU 上，而不是在每个 GPU 上复制整个模型(具体来说， 假设模型`m`包含 10 层：使用`DataParallel`时，每个 GPU 都具有这 10 层中每个层的副本，而当在两个 GPU 上并行使用模型时，每个 GPU 可以承载 5 层）

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10).to('cuda:0')
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to('cuda:1')

    def forward(self, x):
        x = self.relu(self.net1(x.to('cuda:0')))
        return self.net2(x.to('cuda:1'))
```

对于模型太大而无法放入单个 GPU 的情况，上述实现解决了该问题。 但是，分析原理的时候可能已经注意到，如果模型单个GPU可以载入，它将比在单个 GPU 上运行它要慢。 这是因为在任何时间点，两个 GPU 中只有一个在工作，而另一个在那儿什么也没做。 由于中间输出需要在`layer2`和`layer3`之间从`cuda:0`复制到`cuda:1`，因此性能进一步恶化。

除此之外，还有通过异步方式构建流水线的手段加速，这个Trick比较精巧和优雅，思想可以学习一下

```python
class PipelineParallelResNet50(ModelParallelResNet50):
    def __init__(self, split_size=20, *args, **kwargs):
        super(PipelineParallelResNet50, self).__init__(*args, **kwargs)
        self.split_size = split_size

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.seq1(s_next).to('cuda:1')
        ret = []

        for s_next in splits:
            # A. s_prev runs on cuda:1
            s_prev = self.seq2(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

            # B. s_next runs on cuda:0, which can run concurrently with A
            s_prev = self.seq1(s_next).to('cuda:1')

        s_prev = self.seq2(s_prev)
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

        return torch.cat(ret)

setup = "model = PipelineParallelResNet50()"
pp_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
pp_mean, pp_std = np.mean(pp_run_times), np.std(pp_run_times)

plot([mp_mean, rn_mean, pp_mean],
     [mp_std, rn_std, pp_std],
     ['Model Parallel', 'Single GPU', 'Pipelining Model Parallel'],
     'mp_vs_rn_vs_pp.png')
```

在2 GPUs上能有50%的提速，离100%还是有点差距



### 

### 
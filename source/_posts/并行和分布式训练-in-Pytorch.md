---
title: 并行和分布式训练 in Pytorch
toc: true
mathjax: true
top: true
cover: true
date: 2020-11-25 14:54:43
updated:
categories: Pytorch
tags:
	- Pytorch
	- NLP
---

　　虽然还没有机会用到CUDA集群，但是前段时间对协程和并行化的研究，让我忍不住想要探索一下如何在多个GPU下利用Pytorch加快训练的实践方法，算是为之后并行训练优化一个理论参考吧！

　　Pytorch 大体上有3种实现并行的接口（另外还有一种不利用接口的拼接模型的技巧，之后再单独讨论），分别是：nn.DataParallel, nn.multiprocessing, nn.para
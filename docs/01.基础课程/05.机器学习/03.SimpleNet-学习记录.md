---
title: SimpleNet 学习记录
date: 2023-07-05 21:59:55
description: null
tags: 
  - ML
permalink: /pages/09ad82/
categories: 
  - 基础课程
  - 机器学习
author: 
  name: ReturnTmp
  link: https://github.com/ReturnTmp
---





操作系统：windows11

##### 创建虚拟环境

```bash
# 创建
conda create -n simple-net python=3.8
# 删除
conda remove -n simple-net --all
# 激活
conda activate simple-net
# 退出
conda deactivate
```



> 注意：windows 下激活和退出虚拟环境，有时候是不能加conda，自己使用conda env list 尝试，powershell 是需要添加conda的



已经激活虚拟环境安装依赖可以使用如下命令

```bash
pip install XXX
conda install XXX
# 删除命令
conda remove ...
```

未激活可以使用

```bash
conda install -n simple-net XXX
```

查看所有环境

```bash
conda env list
```

##### 查看环境所有依赖

conda list





##### 换源

可以根据清华镜像官网配置：[anaconda | 镜像站使用帮助 | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)





##### 安装依赖

```bash
# 已经激活环境
conda install torchvision==0.13.1 torch==1.12.1 numpy==1.22.4 opencv-python==4.5.1
```











## 前言

论文/代码（paperwithcode）：[SimpleNet: A Simple Network for Image Anomaly Detection and Localization | Papers With Code](https://paperswithcode.com/paper/simplenet-a-simple-network-for-image-anomaly)

论文链接：[2303.15140.pdf (arxiv.org)](https://arxiv.org/pdf/2303.15140.pdf)

代码链接：[DonaldRR/SimpleNet (github.com)](https://github.com/DonaldRR/SimpleNet)

## 论文解读

### 概要

本文提出一个简单和应用友好（application-friendly）的网络( SimpleNet )用于检测和定位异常。SimpleNet由四个组件组成：（1） 生成局部特征的预训练特征**提取器**（pretrained feature **extractor**），（2） 将局部特征传输到目标域（target domain）的浅层特征**适配器**（shallow feature **adapter**），（3） 一个简单的异常特征**生成器**（anomaly feature **generator**），它通过将高斯噪声添加到正常特征来伪造异常特征，以及 （4） 区分异常特征和正常特征的二进制异常**鉴别器**（binary anomaly **discriminator**），用于区分异常特征与正常特征。在推理过程中，将丢弃异常特征生成器（**即异常特征生成器仅在训练期间使用**）。本文的方法基于三种直觉。首先，将预训练的特征转换为面向目标的特征有助于避免领域偏差。其次，在特征空间中生成合成异常更为有效，因为缺陷在图像空间中可能没有太多的共性。第三，一个简单的鉴别器是非常有效和实用的。尽管简单，但SimpleNet在数量和质量上都优于以前的方法。在MVTec AD基准测试中，SimpleNet实现了99.6%的异常检测AUROC，与性能次之的模型相比，误差降低了55.5%。此外，SimpleNet比现有的方法更快，在3080ti的GPU上具有77FPS的高帧率。此外，SimpleNet演示了在One-Class新奇性检测任务上的显著性能改进。

![image-20230727161652204](https://cdn.jsdelivr.net/gh/Returntmp/blog-image@main/blog/image-20230727161652204.png)

### 1. 简介

图像**异常检测和定位任务**旨在识别异常图像并定位异常子区域。检测各种感兴趣的异常现象的技术在工业检测中有着广泛的应用。在工业场景中，异常检测和定位尤其困难，因为**异常样本很少**，异常可能从**细微的变化**（如薄划痕）到大的**结构缺陷**（如缺失零件）不等。

目前的方法主要以**无监督**的方式解决这个问题，即在**训练过程中只使用正常样本**。使用**基于重建的方法（econstruction-based）、基于合成的方法（synthesizing-based）和基于嵌入的方法（embedding-based）**是解决无监督异常检测问题的**三个主要趋势**。

基于重建的方法，假设仅用正常数据训练的深度网络**不能准确地重建异常区域**。将逐像素重建误差作为用于异常定位的异常分数。然而，这一假设可能并不总是成立的，有时网络可以很好地“泛化”，从而也可以很好的重建异常输入，从而导致错误检测。

基于合成的方法通过对在无异常图像上生成的合成异常进行训练来估计正常和异常之间的决策边界。然而，合成的图像不够逼真。来自合成数据的特征可能会偏离正常特征很远，使用这种负样本进行训练可能会导致有松散边界的正常特征空间，这意味着**模糊的缺陷可能会被包括在分布特征空间中**。

最近，基于嵌入的方法实现了最先进的性能。这些方法使用ImageNet预训练的CNN来提取广义正态特征。然后采用多元高斯分布、归一化流和内存库等统计算法嵌入正态特征分布。通过将输入特征与学习的分布或记忆的特征进行比较来检测异常。然而，工业图像通常具有与ImageNet不同的分布。直接使用这些有偏见的特征可能会导致失配问题。此外，统计算法总是受到高计算复杂度或高内存消耗的影响。

为了缓解上述问题，我们提出了一种新型的异常检测和定位网络，称为SimpleNet。SimpleNet利用了基于合成和基于嵌入的方式，并做了一些改进。首先，我们**不直接使用预训练的特征**，而是使用一个特征适配器来产生面向目标的特征，以**减少领域偏差**。第二，我们不直接合成图像上的异常点，而是通过**对特征空间中的正常特征施加噪声来产生异常特征**。我们认为，通过适当地校准噪声的尺度，可以得到一个紧密结合的正常特征空间。第三，我们通过训练一个**简单的判别器**来简化异常检测程序，这比上述基于嵌入的方法所采用的复杂统计算法的计算效率要高得多。具体来说，SimpleNet利用一个预先训练好的骨干来提取正常特征，然后用一个特征适配器将特征转移到目标域。然后，通过向适应的正常特征添加高斯噪声，简单地生成异常特征。一个由几层MLP组成的简单判别器被训练在这些特征上，以判别异常情况。

SimpleNet易于训练和应用，具有出色的性能和推理速度。所提出的SimpleNet基于广泛使用的WideResnet50骨干网，在MVTec AD上实现了99.6%的AUROC，同时运行速度为77fps，在准确性和效率上都超过了之前公布的最佳异常检测方法。我们进一步将SimpleNet引入到单类新颖性检测的任务中，以显示其通用性。这些优势使SimpleNet成为学术研究和工业应用之间的桥梁。代码将公开提供。



### 2. 相关工作

异常检测和定位方法主要可分为三种类型，即基于重建的方法、基于合成的方法和基于嵌入的方法。

**基于重构的方法**认为，异常的图像区域不应该被正确地重构，因为它们不存在于训练样本中。一些方法[10]利用**生成模型**，如**自动编码器和生成对抗网络[11]来编码**和重建正常数据。其他方法[13,21,31]将异常检测作为一个画图问题，图像中的斑块被随机掩盖。然后，利用神经网络来预测被抹去的信息。整合结构相似性指数（SSIM）[29]损失函数被广泛用于训练。异常图被生成为输入图像和其重建图像之间的像素级差异。然而，如果异常点与正常的训练数据有共同的组成模式（如局部边缘），或者解码器对某些异常编码的解码能力 “太强”，那么图像中的异常点就有可能被很好地重建[31]。

**基于合成的方法**通常在无异常的图像上合成异常点。DRÆM[30]提出了一个网络，该网络以端到端的方式对合成的刚出炉的模式进行判别训练。CutPaste[17]提出了一个简单的策略来生成用于异常检测的合成异常点，该策略在大图像的随机位置剪切一个图像补丁并进行粘贴。一个CNN被训练来区分正常和增强的数据分布的图像。然而，合成异常点的外观与真实异常点的外观并不紧密匹配。在实践中，由于缺陷是多种多样且不可预测的，生成一个包括所有异常值的异常集是不可能的。用所提出的SimpleNet代替合成图像上的异常现象，在特征空间中合成负面样本。

**基于嵌入的方法**最近取得了最先进的性能。这些方法将**正常特征嵌入到一个压缩的空间**。**异常特征在嵌入空间中远离正常集群。**典型的方法[6,7,22,24]利用在ImageNet上预先训练好的网络进行特征提取。通过预训练的模型，PaDiM[6]通过多变量高斯分布嵌入提取的异常补丁特征。PatchCore[22]使用名义斑块特征的最大代表存储库。在测试中采用Mahalanobis距离或最大特征距离对输入特征进行评分。然而，工业图像通常具有与ImageNet不同的分布。直接使用预训练的特征可能会造成不匹配的问题。此外，无论是计算协方差的逆值[6]还是通过内存库中的近邻搜索[22]都会限制实时性能，尤其是对于边缘设备。

CS-Flow[24]、CFLOW-AD[12]和DifferNet[23]提出通过归一化流（NF）[20]将**正常特征分布转化为高斯分布**。由于归一化流只能处理全尺寸的特征图，即不允许向下采样，而且耦合层[9]消耗的内存是正常卷积层的几倍，所以这些方法都很耗费内存。蒸馏法[4, 7]训练学生网络以匹配固定的预训练的教师网络的输出，只用正常的样本。在异常查询的情况下，学生和教师的输出之间的差异应该被检测出来。计算的复杂性是双倍的，因为输入图像应该同时通过教师和学生。

SimpleNet克服了上述的问题。SimpleNet使用一个特征适配器，在目标数据集上进行**转移学习**，以减轻预训练的CNN的偏见。SimpleNet建议在**特征空间中合成异常**，**而不是直接在图像上合成**。SimpleNet在推理时遵循**单流方式**，完全由传统的CNN模块构建，这有利于快速训练、推理和工业应用。



### 3. 方法

本节将详细介绍拟议的SimpleNet。如图3所示，SimpleNet由一个特征提取器、一个特征适应器、一个异常特征生成器和一个判别器组成。异常特征生成器只在训练过程中使用，因此SimpleNet在推理过程中采用单流方式。这些模块将在下文中依次描述。

![image-20230727091123692](https://cdn.jsdelivr.net/gh/Returntmp/blog-image@main/blog/image-20230727091123692.png)

#### 3.1 特征提取器

特征提取器获取本地特征，如~\cite{roth2022towards} 中所述。我们将该过程重新表述如下。我们将训练集和测试集分别表示为 $\mathcal{X}{train}$ 和 $\mathcal{X}{test}$。对于 $\mathcal{X}{train}\bigcup\mathcal{X}{test}$ 中的任意图像 $x_{i}\in\mathbb{R}^{H\times W \times 3}$，预训练网络 $\phi$ 从不同的层次中提取特征，通常采用 ResNet 等骨干网络。由于预训练网络对其训练的数据集存在偏差，因此合理的做法是仅选择用于目标数据集的层次子集。形式上，我们定义 $L$ 作为包含要使用的层次索引的子集。来自层次 $l \in L$ 的特征映射表示为 $\phi^{l, i} \sim \phi^{l}(x_{i})\in\mathbb{R}^{H_{l}\times W_{l}\times C_{l}}$，其中 $H_{l}$、$W_{l}$ 和 $C_{l}$ 分别是特征映射的高度、宽度和通道数。对于位于位置 $(h, w)$ 的条目 $\phi^{l,i}{h,w}\in\mathbb{R}^{C{l}}$，其大小为 $p$ 的邻域定义为
$$
\begin{equation} \label{eq1} \small
	\begin{split}
		\mathcal{N}_{p}^{(h,w)} =  \{ (h',y')|
		& h'\in \left [ h-\left \lfloor  p/2\right \rfloor,...,h+\left \lfloor p/2 \right \rfloor  \right ], \\
		& y'\in \left [ w-\left \lfloor  p/2\right \rfloor,...,w+\left \lfloor p/2 \right \rfloor  \right ] \}
	\end{split}
\end{equation}
$$




## 效果展示







## 参考文章

[CVPR 2023 | SimpleNet：一个简单的图像异常检测和定位网络 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/619199955)

[异常检测(无监督)—SimpleNet: A Simple Network for Image Anomaly Detection and Localization](https://blog.csdn.net/weixin_62848630/article/details/130221557?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168999319616800222840190%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168999319616800222840190&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-130221557-null-null.142^v90^control_2,239^v3^insert_chatgpt&utm_term=simplenet&spm=1018.2226.3001.4187)

[更快更准更简单的工业异常检测最新SOTA：SimpleNet - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/642087017)

## 推荐阅读

[简单网络：用于图像异常检测和定位的简单网络 (SimpleNet: A Simple Network for Image Anomaly Detection and Localization)](https://www.zhuanzhi.ai/paper/e6d956988535550fba031c9ad18ed9fd)

[2023CVPR SimpleNet A Simple Network for Image Anomaly Detection and Localizations](https://blog.csdn.net/Vincent_Tong_/article/details/130414293)
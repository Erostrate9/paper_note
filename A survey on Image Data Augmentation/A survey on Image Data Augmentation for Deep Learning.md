[TOC]

# A survey on Image Data Augmentation for Deep Learning

## Keywords

Data Augmentation

Big data

Image data

Deep Learning

GANs

## Abstract

### Image augmentation algorithms:

* geometric transformations
* color space augmentations
* kernel filters
* mixing images
* random erasing
* feature space augmentation
* adversarial training
* generative adversarial networks
* neural style transfer
* meta-learning

**The application of augmentation methods based on GANs are heavily covered in this survey.**

### other characteristic of Data Augmentation tech

* test-time augmentation

* resolution impact

* final dataset size

* curriculum learning

## Introduction

* Deep neural networks has been successfully applied to Computer Vision tasks. e.g: image classification, object detection, and image segmentation. convolutional neural networks(CNNs) 的成功激起了人们对将深度学习应用于计算机视觉任务的兴趣和乐观态度。

  ​	* CNN （**Convolutional Neural Network**/ConvNet）

## Image Data Augmentations techniques

### Geometric versus photometric transformations

#### Kernel filters

#### Mixing images

#### Random erasing

#### A note on combining augmentations

### Data Augmentations based on Deep Learning

#### Feature space augmentation

Feature space:  Lower-dimensional representations found in high-level layers of a CNN are known as the feature space.

CNN中低维的原始数据被映射到高维的空间，特征空间中的特征是对原始数据更高维的抽象。

* SMOTE
  * class imbalance: 分类时一种类的样本显著多于另一类。
  * Naive方法：简单拷贝少数类，不会给模型提供额外的信息。
  * SMOTE（**Synthetic Minority Oversampling Technique**）： 一种解决class imbalance的增强方式。扩增（综合现有样本，合成新的样本）少数类minority class的样本数量。https://arxiv.org/abs/1106.1813
  * 这种oversample技术通过连接k个最近的邻居来形成新的实例，从而应用于特征空间。
  * 选择特征空间中相近的例子a和b，在ab之间画一条线，并在沿该线的某一点画一个新的样本。*synthetic instances*被作为a和b的一个凸组合(*convex combination*)生成。
  * 一个有效的方案：将SMOTE与多数类的under-sampling相结合。
  * 缺点：不适用于有大量重叠的二元分类。
* DeVries  and  Taylor 讨论了对特征空间的增强，为数据增强的向量操作提供了机会。DeVries 和 Taylor 讨论了添加噪声、内插和外推作为特征空间增强的常见形式。
   * adding noise
   * interpolating
   * extrapolating

*   auto-encoders 
  * encoder将原始图像映射到低维向量表示（encoded representation）。encoded representation用于数据增强。
  * decoder将向量重建回原始图像

#### Adversarial training

* 对抗性训练是一个使用两个或更多的网络的框架，这些网络的损失函数中编码了相反的目标。本节讨论了使用对抗性训练作为一种搜索算法，以及对抗性攻击的现象。对抗性攻击由一个对手网络组成，该网络学习对图像的增强，导致其对手分类网络的错误分类。
* 对抗性攻击揭示了一个反直觉的事实：图像的表示远不如预期的那么健壮。
* 方案1：FGSM，PGD（Projected Gradient Descent），对抗网络学习导致对手网络错误分类的最小插入噪声。并不会提高分类网络的分辨准确率，但是会增强网络的鲁棒性，抗干扰。
* 方案2：DistrubLabel，打乱标签，是一种正则化技术(regularization)，可以在每次迭代时随机替换标签。这是向损失层添加噪声的罕见例子，而大多数其他增强方法都向Input 或 hidden presentation layers添加噪声。

#### GAN-based Data Augmentation
* Generative modeling: 生成建模是指从数据集创建人工实例的做法，以便它们保留与原始集相似的特征。
* GANs: Generative adversarial network	生成对抗网络
  * GANs can be seen as a way to "unlock" additional information from a dataset.
  * GANs是一种速度快、质量好的Generative modeling技术。
  * 在Data augmentation中最有前途的generative modeling技术
* VAE: variational auto-encoders 变分自编码器
  * 可以扩展 GAN 框架以提高使用变分自动编码器生成的样本的质量
  * VAE像先前讨论的特征空间增强一样，学习数据点的低维表示，
  * Generator network就像造假币的counterfeiter，Discriminator就像警察（验钞机）。
  * 博弈论表明the generator will eventually fool the discriminator.

<img src="https://raw.githubusercontent.com/Erostrate9/img/main/20211128144315.png" style="zoom:70%;" />

为了扩展GANs的概念和产生更高分辨率的输出图像，已经提出了许多新的体系结构，例如: **DCGANs、Progressively Growing GANs, CycleGANs, and Conditional GANs** 似乎在数据增强方面具有最大的应用潜力。

* DCGAN: 使用CNN而不是感知生成器，扩展生成器和鉴别器网络的内部复杂性(internal complexity)；使用反卷积层(deconvolutional layers)，扩展图像空间维度。’
  * medical term: *Sensitivity*: the ability of a test to correctly identify patients with a disease. *Specificity*: the ability of a test to correctly identify people without the disease. *True positive*: the person has the disease and the test is positive. *True negative*: the person does not have the disease and the test is negative.
* Progressively Growing GANs
  * 将样本从低分辨率的GAN传递到高分辨率的GAN
  * facial images表现出色

* CycleGAN
   * *Cycle-Consistency loss function*
   * CycleGAN学会了从一个图像域迁移到另一个域，例如马到斑马。借助forward and backward consistency loss functions.
   * 生成器A: 马 -> (fake)斑马
   * 鉴别器A: 无法判断fake斑马是否是斑马集的一部分
   * 生成器B: (fake)斑马 -> (re)马
   * 鉴别器B:  判断(re)马是否属于马集合.
   * 鉴别器A.loss+鉴别器B.loss -> cycle-consistency loss
   * CycleGAN可以用作智能的过采样解决类不平衡问题.
   * <img src="https://raw.githubusercontent.com/Erostrate9/img/main/20211129103526.png" style="zoom:50%;" />
   * 为了进一步了解添加 GAN 生成实例的有效性，使用了 t-SNE 可视化。 t-SNE是一种可视化技术，它学习将高维向量之间映射到低维空间，以促进决策边界的可视化
* Conditional GANs
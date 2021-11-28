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


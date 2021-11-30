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

### Data Augmentations based on basic image manipulations

#### Geometric transformations

#### Flipping

#### Color space

#### Cropping

#### Rotation

#### Translation

#### Noise injection

#### Color space transformations

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
  * 条件GANs在生成器和鉴别器上都增加了一个条件向量，以缓解模态崩溃的问题。
* GANs的缺点：
  * 1. 从目前的尖端架构中获得高分辨率的输出是非常困难的。增加生成器产生的图像的输出大小将可能导致训练不稳定和不收敛。
  * 2. 需要大量数据来训练。

#### Neural Style Transfer

* 神经风格迁移因其艺术应用而闻名，亦可用作数据增强的工具。
* manipulating the sequential representations across a CNN
* 保留图片内容的同时迁移另一张图片的风格。
* Fast Style Transfer
  * 该算法将损失函数从每像素损失扩展为感知损失，并使用前馈feed-forward网络对图像进行样式化stylize。
  * perceptual loss (感知损失): 将真实图片卷积得到的feature与生成图片卷积得到的feature作比较，使得高层信息（内容和全局结构）接近，也就是感知的意思。
  * 风格转换中perceptual loss显示出巨大应用前景.
  * 用实例规范化(instance normalization)代替批规范化(batch normalization) 归一化的结果是对快速风格化的显著改善。
    * BN层的正则化作用：在BN层中，每个batch计算得到的均值和标准差是对于全局均值和标准差的近似估计，这为我们最优解的搜索引入了随机性，从而起到了正则化的作用。
    * BN的缺陷：带有BN层的网络错误率会随着batch_size的减小而迅速增大，当我们硬件条件受限不得不使用较小的batch_size时，网络的效果会大打折扣。LN，IN，GN就是为了解决该问题而提出的。
    * ![preview](https://pic2.zhimg.com/v2-66b2a13334967dc27025e354bb448875_r.jpg)
    * 这张图与我们平常看到的feature maps有些不同，立方体的3个维度为别为batch/ channel/ HW，而我们常见的feature maps中，3个维度分别为channel/ H/ W，没有batch。分析上图可知：BN计算均值和标准差时，固定channel(在一个channel内)，对HW和batch作平均；LN计算均值和标准差时，固定batch(在一个batch内)，对HW和channel作平均；IN计算均值和标准差时，同时固定channel和batch(在一个batch内中的一个channel内)，对HW作平均；GN计算均值和标准差时，固定batch且对channel作分组(在一个batch内对channel作分组)，在分组内对HW作平均
  * 风格迁移的应用：1. 艺术 2. 自动驾驶，将训练数据迁移到昼/夜、不同天气
  * 一个矛盾：到底是利用GANs让训练数据更真实好还是用风格迁移让数据有更好地风格多样性好？
  * 缺陷：如果样式集太小，可能引入更多偏差。其中一种算法需要大量内存和计算，运行慢，不适用于数据增强；另一种转换限制在预训练集上。

#### Meta learning Data Augmentations

* 用神经网络优化神经网络。
* 其他方案
  * evolutionary algorithm for architecture search
  * 简单随机搜索
  * 以上二者结合
* 本文讨论的meta-learning模式: 神经网络的，基于梯度
  * 元架构设计是下一个范式转变
  * 用prepended神经网络通过混合图像、神经风格迁移和几何变换来学习数据增强
* Neural augmentation
  * 一种元学习神经风格迁移策略算法
  * 接收来自同一类的两个随机图像。预先设计好的增强网络通过CNN将它们映射成一个新的图像。扩增输出的图像与另一个随机图片神经风格迁移
  * 通过CycleGAN实现
  * 最好的策略可能是将传统的增强和神经增强结合起来
* Smart Augmentation
  * 与Neural augmentation类似，但是在图像的组合完全来自预先设置的CNN的学习参数。
  * 通过两个网络实现。网络A是一个增强网络，接受两个或多个输入图像，将它们映射到一个或多个新图像，以训练网络B；网络B错误率的变化被反向传播，以更新网络A。网络A包含另一个损失函数，以确保其输出与类内其他图像相似。网络A使用一系列的卷积层来产生增强图像。
  * 网络A可以被扩展成多个并行训练的网络。
  * SA也是组合现有的实例产生新的实例，但比传统的SamplePairing或mixed-examples复杂得多。SA使用一个自适应的CNN来产生新的图像实例。
* AutoAugment
  * 一种强化学习算法.
  * 在具有各种扭曲程度的几何变换的限制性集合中寻找最佳的增强policy。
  * policy: 在强化学习中，policy类似于学习算法的策略，这个policy决定了在给定状态下为了实现某些目标要采取什么行为。
  * AutoAugment学习一个由多个sub-policy组成的policy。每个sub-policy包括图像转换和转换幅度。因此强化学习被用做一个增强的离散搜索算法。
  * 作者认为进化算法或随机搜索也会是有效的搜索算法。
  * 有研究认为由于离散搜索带来的明显缺陷，强化学习搜索算法劣于Augmented Random Search。
  * Minh等人尝试使用强化学习来搜索数据增强，针对单个实例而不是整个数据集的学习迁移。使用强化学习增强搜索训练的模型在测试时间增强方面鲁棒性更好
* 元学习的一个缺点是它是一个相对较新的概念，并没有经过严格的测试。此外，元学习方案可能难以实施，且耗时。

### Comparing Augmentations

* compared GANs, WGANs, flipping, cropping, shifting, PCA jittering, color jittering, adding noise, rotation, and some combinations on the CIFAR-10 and ImageNet datasets, cropping, flipping, WGAN, and rotation generally performed better than others. The combinations of flipping + cropping and flip- ping + WGAN were the best.

## Design considerations for image Data Augmentation

### Test-time augmentation

* TTA是对测试数据集的增强，可获得更鲁棒的预测。
* 对医疗图像诊断很有价值。
* TTA对分类准确性的影响是另一种衡量分类器鲁棒性的机制。一个鲁棒的分类器其预测结果应受TTA影响较小。例如，当测试数据旋转一定角度，分类器还应具备正确的识别能力。
* 考虑到速度必要性，一些分类模型不能做出决定，我们可以考虑一种逐步提高预测置信度的办法：首先输出很少或没有TTA的预测，然后增量地添加TTA来增加预测的置信度。

### Curriculum learning

* 有研究者认为按某种顺序输入训练数据比无筛选的随机选择输入要好

  * 例如，有人认为最好先只用原始数据训练，然后用原始数据+增强数据完成训练。这一点目前还没有共识。
  * 有研究者认为只用增强数据进行训练获取的初识权重和从其他数据集迁移得到的权重类似，之后只需用原始训练数据微调。

  * Curriculum learning对One-Shot学习很重要，例如FaceNet，找到与新面孔类似的面孔很重要。
  * 数据增强从翻转、翻译和随机擦除等组合中构建了大规模膨胀的训练。很可能在这个集合中存在一个子集，利用它训练将更快、更准确。

### Resolution impact

* 直觉告诉我们，未来的模型期望能从高分辨率图像输入下进行训练，因为对原始图像下采样极有可能导致信息丢失。
* 组成一个由高分辨率和低分辨率图像训练的模型集合比任何一个单独的模型表现更好。因此，不同的下采样图像可以被视为另一种数据增强方案。
  * 同样，上采样作为DA也是可取的。
* 分辨率对GANs也十分重要。涉及：训练稳定性，模式崩塌
  * mode collapse: Mode collapse happens **when the generator can only produce a single type of output or a small set of outputs**. This may happen due to problems in training, such as the generator finds a type of data that is easily able to fool the discriminator and thus keeps generating that one type.
  * 目标：从GAN samples中有效生成高分辨率输出

### Final dataset size

* 增强数据直接并上原数据集后需要额外内存和算力。
  * 可以选择使用生成器在训练期间动态转换数据，或者事先转换数据并将其存储在内存中
  * 动态转换数据可以节省内存，但会导致较慢的训练。
  * 根据数据集大小膨胀的程度，在内存中存储数据集可能会产生极大的问题。在增强大数据时，在内存中存储增强的数据集尤其成问题。此决策通常分为在线online或离线offline数据增强（在线增强指动态增强，离线增强指在磁盘上编辑和存储数据）。
* 在大规模分布式训练系统的设计中，可以在训练前对图像进行增强，以加速图像服务。通过预先增强图像，分布式系统能够请求和预缓存训练批。增强也可以构建到计算图中，用于构建深度学习模型并促进快速区分。
* 探索膨胀数据的一个子集也很有趣，这将导致与整个训练集更高或相似的性能。这是一个类似于课程学习的概念，因为其核心思想是找到训练数据的最优顺序。这个想法也与最终数据集大小、转换计算和存储增强图像的可用内存非常相关。

### Alleviating class imbalance with Data Augmentation

* a naive solution:  oversampling minority class. with simple image manipulations.
  * problem: 过拟合
* Oversampling methods based on Deep Learning such as 对抗训练，神经风格迁移，GANs，元学习模式
  * NST是一个创建新图像的有趣方法。
  * GANs 有效增加少数类大小同时保留外在分布。
    * a. 用整个少数类作为real examples
    * b. 用少数类的子集作为GANs的输入。
      * 未来展望：evolutionary sampling进化采样获取子集.

## Discussion

* two general categories to augment image data:
  * data warping
  * oversampling
* 有些augmentation方法很直观，有些很反直觉 ( e.g. mixing)
* future work: 通过主要探索test-time几何变换和NST来确定ttA的有效性
* future work: 确定增强后数据集大小多大合适。增强太多显然不合适，会导致过拟合。
* 目前还没有增强技术可以纠正关于测试数据的多样性非常差的数据集。在训练数据和测试数据都来自相同分布的假设下，所有这些增强算法都表现最好，但如果不是 这些方法可能就没用了。

## Future work

* such as: 建立增强技术的分类法，提高GAN样本的质量，学习元学习和数据增强相结合的新方法，发现数据增强和分类器架构之间的关系，并将这些原则扩展到其他数据类型。视频数据中的时间序列成分如何影响静态图像增强技术的使用。image DA技术向其他领域的迁移应用。
* 提高GAN样本的质量并在广泛的数据集上测试其有效性
* Super-resolution networks through the use of SRCNNs, Super-Resolution Convolutional Neural Networks, and SRGANs
* ttA
* Meta-learning GAN architectures
  * 在生成器和鉴别器架构上使用强化学习算法（如NAS）
  * 使用一个evolutionary approach 通过并行化和集群计算加快GANs的训练速度
* 开发软件工具. DA libraries
  * Keras: ImageDataGenerator
  * Albumentations

## Conclusion

## Abbreviations
GAN: generative adversarial network

CNN: convolutional neural network

DCGAN: deep convolutional generative adversarial network; 

NAS: neural architecture search; 

SRCNN: super-resolution convolutional neural network; 

SRGAN: super-resolution generative adversarial network;

CT: computerized tomography;

MRI: magnetic resonance imaging;
PET: positron emission tomography; 

ROS: random oversampling; 

SMOTE: synthetic minority oversampling technique;
RGB: red-green–blue; 

PCA: principal components analysis; 

UCI: University of California Irvine; 

MNIST: Modified National Institute of Standards and Technology; 

CIFAR: Canadian Institute for Advanced Research; 

t-SNE: t-distributed stochastic neighbor embedding.


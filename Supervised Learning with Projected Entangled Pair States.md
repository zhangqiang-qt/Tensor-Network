# Supervised Learning with Projected Entangled Pair States

> 这份笔记是在阅读论文《Supervised Learning with Projected Entangled Pair States》的过程中所做的一些记录和补充。论文的出处以及阅读过程中产生的其余文献的参考，均置于文末参考文献中。

> 注：这篇论文已经发表在PRB上，根据配图来看，正式发表的论文相比于arXiv上的预印版，应该有所补充或完善。但是，发表的论文虽然有了DOI号，不知道是不是还没有被SCI检索的原因，暂时无法下载。

## 一、研究的问题
&emsp;&emsp;目前，基于张量网络的机器学习方法，在进行数据处理的时候，总是考虑将其表示为近似一维的与树结构类似的张量网络，比如矩阵乘积态(Matrix Product State, MPS)、树张量网络(Tree Tensor Network, TTN)等。当然，在实际的应用中也取得了不错的效果。但是，基于MPS和TTN的模型在处理图像的时候，邻近像素点之间的结构和空间关联被忽略，并且导致短程关联被人为地变成长程关联，带来了不必要的计算开销和统计偏差。而投影纠缠对态(Projected Entangled Pair States, PEPS)具有与原始图像类似的二维结构，因此，论文考虑利用PEPS来建立机器学习模型。

## 二、PEPS图像分类
### 1. 输入数据的特征映射
&emsp;&emsp;监督学习的目的就是寻找一个函数或者映射 <img src="https://latex.codecogs.com/png.latex?f(x)" title="f(x)" />，将输入的训练图像 <img src="https://latex.codecogs.com/png.latex?x&space;\in&space;\mathbb{R}^{L_{0}&space;\times&space;L_{0}}" title="x \in \mathbb{R}^{L_{0} \times L_{0}}" /> 映射为 <img src="https://latex.codecogs.com/png.latex?y&space;\in&space;\{1,&space;2,&space;\cdots,&space;T\}" title="y \in \{1, 2, \cdots, T\}" />。由于非线性性能够有效提升输入空间的维度，使得提取数据的特征更加容易，因此这样的映射通常是高度非线性的。而张量网络是能够作用在高维空间中的线性模型，于是所有的特征就变得线性可分，不需要考虑非线性性。因此，我们首先需要将输入数据 <img src="https://latex.codecogs.com/png.latex?x" title="x" /> 转换为特征张量 <img src="https://latex.codecogs.com/png.latex?\Phi(x)" title="\Phi(x)" />，有如下两种特征映射方式：

#### 1) 直积态
&emsp;&emsp;将黑色像素点 <img src="https://latex.codecogs.com/png.latex?x_{i}&space;=&space;0" title="x_{i} = 0" /> 表示为态 <img src="https://latex.codecogs.com/png.latex?|0\rangle=\left[\begin{array}{l}1&space;\\&space;0\end{array}\right]" title="|0\rangle=\left[\begin{array}{l}1 \\ 0\end{array}\right]" />，白色像素点 <img src="https://latex.codecogs.com/png.latex?x_{i}&space;=&space;1" title="x_{i} = 1" /> 表示为态 <img src="https://latex.codecogs.com/png.latex?|1\rangle=\left[\begin{array}{l}0&space;\\&space;1\end{array}\right]" title="|1\rangle=\left[\begin{array}{l}0 \\ 1\end{array}\right]" /> 。那么，介于0和1之间的像素值可以表示为如下的叠加态：

<p align="center"><img src="https://latex.codecogs.com/png.latex?\phi\left(x_{i}\right)=a|0\rangle&plus;b|1\rangle=a\left[\begin{array}{l}&space;1&space;\\&space;0&space;\end{array}\right]&plus;b\left[\begin{array}{l}&space;0&space;\\&space;1&space;\end{array}\right]" title="\phi\left(x_{i}\right)=a|0\rangle+b|1\rangle=a\left[\begin{array}{l} 1 \\ 0 \end{array}\right]+b\left[\begin{array}{l} 0 \\ 1 \end{array}\right]" /></p>

其中 <img src="https://latex.codecogs.com/png.latex?a=\cos&space;\frac{\pi&space;x_{i}}{2}" title="a=\cos \frac{\pi x_{i}}{2}" />，<img src="https://latex.codecogs.com/png.latex?b=\sin&space;\frac{\pi&space;x_{i}}{2}" title="b=\sin \frac{\pi x_{i}}{2}" />。那么，对于图像 <img src="https://latex.codecogs.com/png.latex?N&space;=&space;L_{0}&space;\times&space;L_{0}" title="N = L_{0} \times L_{0}" /> ，其特征张量 <img src="https://latex.codecogs.com/png.latex?\Phi(x)" title="\Phi(x)" /> 定义为：

<p align="center"><img src="https://latex.codecogs.com/png.latex?\Phi(x)=\phi\left(x_{1}\right)&space;\otimes&space;\phi\left(x_{2}\right)&space;\otimes&space;\cdots&space;\otimes&space;\phi\left(x_{N}\right)" title="\Phi(x)=\phi\left(x_{1}\right) \otimes \phi\left(x_{2}\right) \otimes \cdots \otimes \phi\left(x_{N}\right)" /></p>

#### 2) 卷积特征映射
&emsp;&emsp;这里的卷积指的是神经网络中的卷积层。卷积层的输入是原始图像 <img src="https://latex.codecogs.com/png.latex?x&space;\in&space;\mathbb{R}^{L_{0}&space;\times&space;L_{0}}" title="x \in \mathbb{R}^{L_{0} \times L_{0}}" />，输出是一个维度为 <img src="https://latex.codecogs.com/png.latex?L&space;\times&space;L&space;\times&space;d" title="L \times L \times d" /> 的三阶特征张量，其中 <img src="https://latex.codecogs.com/png.latex?L&space;\times&space;L(L&space;\leq&space;L_{0})" title="L \times L(L \leq L_{0})" /> 是特征的输出尺寸，<img src="https://latex.codecogs.com/png.latex?d" title="d" /> 表示通道数。不难看出，卷积神经网络的特征映射输出的结果依旧可以看作是一个直积态，每个值都在 <img src="https://latex.codecogs.com/png.latex?L&space;\times&space;L" title="L \times L" /> 大小的格点上，并且物理维度为 <img src="https://latex.codecogs.com/png.latex?d" title="d" />。因此，特征张量的总的空间大小为 <img src="https://latex.codecogs.com/png.latex?d^{L&space;\times&space;L}" title="d^{L \times L}" />。
传统的多层感知器(MLP)是将特征张量 <img src="https://latex.codecogs.com/png.latex?\Phi(x)" title="\Phi(x)" /> 展平为一个向量作为输入，忽略了特征张量的空间结构。在这里，考虑一种使用二维张量网络的线性分类器，其可以充分地将原始特征张量作为输入，这样可以保留空间结构。

> 关于多层感知器(MLP)，YouTobe大佬3B1B讲解的很棒，视频链接附于文末。

### 2. PEPS分类器
&emsp;&emsp;考虑一个线性映射 <img src="https://latex.codecogs.com/gif.latex?W" title="W" />，得到一个向量，表示给定输入图像的 <img src="https://latex.codecogs.com/gif.latex?T" title="T" /> 个标签之一的概率，即：

<p align="center"><img src="https://latex.codecogs.com/gif.latex?f(x)&space;=&space;W&space;\cdot&space;\Phi(x)" title="f(x) = W \cdot \Phi(x)" /></p>

其中，<img src="https://latex.codecogs.com/gif.latex?W&space;\in&space;\mathbb{R}^{d^{L&space;\times&space;L}&space;\times&space;T}" title="W \in \mathbb{R}^{d^{L \times L} \times T}" />，· 表示 <img src="https://latex.codecogs.com/gif.latex?W" title="W" /> 与 <img src="https://latex.codecogs.com/gif.latex?\Phi(x)" title="\Phi(x)" /> 之间的张量缩并。由于总的参数量达到了 <img src="https://latex.codecogs.com/gif.latex?d^{L&space;\times&space;L}&space;\times&space;T" title="d^{L \times L} \times T" />，利用PEPS对 <img src="https://latex.codecogs.com/gif.latex?W" title="W" /> 进行近似表示：

<p align="center"><img src="https://latex.codecogs.com/gif.latex?W^{l,&space;s_{1}&space;s_{2}&space;\cdots&space;s_{N}}=\sum_{\sigma_{1}&space;\sigma_{2}&space;\cdots&space;\sigma_{K}}&space;T_{\sigma_{1},&space;\sigma_{2}}^{s_{1}}&space;T_{\sigma_{3},&space;\sigma_{4},&space;\sigma_{5}}^{s_{2}}&space;\cdots&space;T_{\sigma_{k},&space;\sigma_{k&plus;1},&space;\sigma_{k&plus;2},&space;\sigma_{k&plus;3}}^{s_{i},&space;l}&space;\cdots&space;T_{\sigma_{K-1},&space;\sigma_{K}}^{s_{N}}" title="W^{l, s_{1} s_{2} \cdots s_{N}}=\sum_{\sigma_{1} \sigma_{2} \cdots \sigma_{K}} T_{\sigma_{1}, \sigma_{2}}^{s_{1}} T_{\sigma_{3}, \sigma_{4}, \sigma_{5}}^{s_{2}} \cdots T_{\sigma_{k}, \sigma_{k+1}, \sigma_{k+2}, \sigma_{k+3}}^{s_{i}, l} \cdots T_{\sigma_{K-1}, \sigma_{K}}^{s_{N}}" /></p>

其中，`$K$`是虚拟指标`$\sigma_{k} \in \{1,2,\cdots,D \}$`的个数。每个张量都有一个物理指标`$s_{i} \in \{1,2,\cdots,d\}$`，与输入向量`$\Phi(x_{i})$`相关联。在中心张量处，有一个多余的标签索引`$l \in \{1,2,\cdots,T\}$`，产生模型的输出向量。这些张量的值随机初始化为`$0$`到`$0.01$`的实数，构成了模型的可训练参数`$θ$`。

<center>

![peps监督学习模型.png](https://note.youdao.com/yws/res/7/WEBRESOURCE1cc5c766f542ba9edca90d3616b6c927)

</center>

### 3. 训练算法
&emsp;&emsp;**训练的目的是**调整参数`$\theta$`，使训练标签和预测标签之间的差值达到最小。通过最小化表示预测标签分布与对应训练标签分布的`$one-hot$`向量之间距离的损失函数`$\mathcal{L}$`来实现，其定义如下：
```math
\mathcal{L}=-\sum_{x_{i}, y_{i} \in \mathcal{T}} \log \left[\operatorname{softmax}\left(f^{\left[y_{i}\right]}\left(\boldsymbol{x}_{i}\right)\right)\right]
```

```math
\operatorname{softmax}\left[f^{\left[y_{i}\right]}(\boldsymbol{x})\right] \equiv \frac{\exp f^{\left[y_{i}\right]}\left(\boldsymbol{x}_{i}\right)}{\sum_{\ell=1}^{T} \exp f^{[\ell]}\left(\boldsymbol{x}_{i}\right)}
```
其中，`$x_{i}$`表示第`$i$`张图片，`$y_{i}$`表示数据集`$\mathcal{T}$`中相应的标签。激活函数`$\operatorname{softmax}$`的输出可以解释为图片`$x_{i}$`属于`$y_{i}$`的概率，损失函数`$\mathcal{L}$`表示模型概率与图像标签的交叉熵。`$f^{[\ell]}$`由PEPS模型的物理指标和特征映射向量缩并得到：
```math
\begin{aligned}
f^{[\ell]}(\boldsymbol{x}) &=W^{\ell, s_{1} s_{2} \cdots s_{N}} \cdot \phi_{s_{1}}\left(x_{1}\right) \otimes \phi_{s_{2}}\left(x_{2}\right) \otimes \cdots \otimes \phi_{s_{N}}\left(x_{N}\right) \\
&=\sum_{\sigma_{1} \sigma_{2} \cdots \sigma_{K}} M_{\sigma_{1}, \sigma_{2}} M_{\sigma_{3}, \sigma_{4}, \sigma_{5}} \cdots M_{\sigma_{k}, \sigma_{k+1}, \sigma_{k+2}, \sigma_{k+3}}^{\ell} \cdots M_{\sigma_{K-1}, \sigma_{K}}
\end{aligned}
```
其中，`$M=\sum_{s_{i}} T^{s_{i}} \phi_{s_{i}}\left(x_{i}\right)$`。

&emsp;&emsp;当PEPS模型比较小的时候，其张量缩并的计算复杂度正比于`$D^{L}$`。当`$D$`和`$L$`很大的时候，需要采用近似方法——边界MPS方法。

#### 边界MPS方法
&emsp;&emsp;我们已经提及，PEPS是一个具有二维结构的张量网络，因此，将位于最底层的一行张量看作MPS，剩下的所有行视为作用在该MPS上的算符。当每一行作用在MPS上时，需要将MPS的键维裁剪为最大值`$\chi$`。而为了获得最小的裁剪误差，首先对MPS使用QR分解，以此确保MPS是正确的正则形式(canonical form)。然后，再对正则化MPS的中心张量使用SVD分解，确保裁剪是最优的。

<center>

![bmps.png](https://note.youdao.com/yws/res/b/WEBRESOURCEdedb1c75e6d16c5781a9d0cf9d4a142b)

</center>

&emsp;&emsp;由边界MPS方法实现的近似缩并的总的计算复杂度为`$O(N\chi^{3}D^{6})$`。更有效的方法是从顶层MPS和底层MPS对PEPS进行并行缩并，此时的计算复杂度为`$O(T\chi^{3}D^{2})$`。

<center>

![边界mps缩并.png](https://note.youdao.com/yws/res/9/WEBRESOURCE068b18a50e836a99fc05ca8c1932ff49)

</center>

&emsp;&emsp;除了预测标签和评估损失函数的正向过程外，还需要逆向过程来计算损失函数相对于训练参数的梯度：利用**张量网络的自动微分技术**。自动微分技术的关键在于将张量网络算法视为关于张量和代数操作的可追踪计算图，然后通过简单的链式法则，可以沿着这个计算图实现反向传播过程，得到损失函数`$\mathcal{L}$`对各个参数的梯度值(就是冉仕举视频中的求梯度的那种方式)。最后，直接使用随机梯度下降和Adam优化器来更新可训练参数`$\theta$`。如果参数均为非负值，优化的稳定性能得到很大的改善。

## 三、实验结果
&emsp;&emsp;关于实验结果，需要关注的是对于初始图像的处理。如果采用的是直积态的方式，那么将`$28\times28$`的图像以`$2\times2$`的块大小与`$14\times14$`的PEPS中的张量进行缩并。以`$6\times6$`的图像为例(如下图所示)，每个像素值均表示为二维向量的形式。对`$2\times2$`的块大小(左上角)，求其张量积得到直积态与PEPS中的张量进行缩并，以此类推。

<center>

![peps.png](https://note.youdao.com/yws/res/7/WEBRESOURCEa61f8e92d374920465f3eb20d8cdb187)

</center>

&emsp;&emsp;第二种方式是利用卷积神经网络对图像进行处理，然后将处理后的数据同样地与PEPS进行缩并：利用`$10$`个大小为`$5\times5$`的卷积核以步长为`$1$`对`$28\times28$`的图像进行卷积操作(需要将原始图像补`$0$`为`$32\times32$`的图像)，每一次卷积都能得到`$28\times28$`的图像，然后经过非线性激活层和`$2\times2$`的最大池化层之后，得到`$14\times14$`的输出。经过`$10$`次这样的卷积操作之后，最终得到`$14\times14\times10$`的三阶张量。同样地，取`$2\times2$`的块大小与`$14\times14$`的PEPS中的张量进行缩并。与方式一的不同之处在于：方式一中PEPS的物理指标维度为`$16$`
，而卷积之后的物理指标维度为`$10$`。两种方式中，PEPS的键维均设置为`$\chi=10$`。

<center>

![cnnpeps.png](https://note.youdao.com/yws/res/1/WEBRESOURCE2732b3200704fcf5cdfc34e127385501)

</center>

&emsp;&emsp;数据集采用的是经典的MNIST和Fashion-MNIST数据集，其包含的具体数据如下图所示。

<center>

![mnist and fashionmnist.png](https://note.youdao.com/yws/res/0/WEBRESOURCEf9565d0e840bdbde9854ae78e9e919a0)

</center>

将论文中提到的两种方式分别在MNIST数据集中进行训练与测试，可以看出，PEPS和CNN-PEPS的准确度明显比MPS结构好很多。键维`$D=4$`时，PEPS的测试准确度超过了传统的MLP；键维`$D=2$`时，CNN-PEPS的测试准确度也超过了CNN-MLP。并且，键维`$D=5$`时，两者达到最优的准确度。

<center>

![mnist准确率.png](https://note.youdao.com/yws/res/7/WEBRESOURCE6749be122f3f0fe086439bff78c61497)

</center>

而在Fashion-MNIST数据集中，相比于MPS、PEPS和MPS结合TTN这三种张量网络结构，CNN-PEPS具有更好的效果。

<center>

![fashionmnist准确率.png](https://note.youdao.com/yws/res/9/WEBRESOURCEbdbe7f5f30f9764800a3bccfe6efb219)

</center>

## 论文中的疑问
- 边界MPS方法对PEPS进行收缩的时间复杂度`$O(N\chi^{3}D^{6})$`，和同时收缩时的时间复杂度`$O(T\chi^{3}D^{2})$`，不清楚是如何计算出来的；
- 为什么特征映射采用直积态的形式，并且保持参数为非负值，能够极大地改善优化的稳定性？论文中并没有给出解释。

### 参考文献
#### 1) 论文

[1] Cheng S, Wang L, Zhang P. Supervised Learning with Projected Entangled Pair States[J]. arXiv preprint arXiv:2009.09932, 2020.

[2] Vidal G. Efficient classical simulation of slightly entangled quantum computations[J]. Physical review letters, 2003, 91(14): 147902.

#### 2) 多层感知器(MLP)
[1] [YouTube](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=1)

[2] [BiliBili](https://www.bilibili.com/video/BV1bx411M7Zx)

#### 3) 交叉熵
[1] [简单的交叉熵，你真的懂了吗？](https://zhuanlan.zhihu.com/p/61944055)

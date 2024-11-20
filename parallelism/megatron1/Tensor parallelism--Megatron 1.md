# Tensor parallelism --Megatron 1

## Contributions

- 实现了 **层内** 张量并行，可以训练更大的模型（简单、高效）。
- 只需要在代码中插入几行 **通信原语** 。
- BERT--like  model 通过改变 layer normalization的位置可以提高准确率。

## MLP block

![megatron1-mlp](/Users/guokunhao/笔记/parallelism/megatron1/megatron1-mlp.png)

- 先 **列切割** ，后 **行切割** 

- f  和  g  是共轭的，f  的前向传播直接复制数据， 反向传播做一次 All_Reduce, d  的前向传播做一次 All_Reduce，反向传播直接复制数据

- f  operator 的例子

  ```python
  class f(torch.autograd.Function):
      def forward(ctx, x):
          return x
      def backward(ctx, gradient):
          all_reduce(gradient)
          return gradient
  ```

  

#### 通信量

前向 和 反向 各需一次 All_Reduce

## Self Attention Block

![megatron1-attention](/Users/guokunhao/笔记/parallelism/megatron1/megatron1-attention.png)

- 利用多头注意力先天的并行优势，在 Q\K\V 矩阵上按**列切割**，在输出矩阵上按**行切割** 。

#### 通信量

前向 和 反向 各需一次 All_Reduce

## Embedding Block（输入层）

![embedding](/Users/guokunhao/笔记/parallelism/megatron1/embedding.png)

- 输入层和输出层共享权重，在 词表 方向切割矩阵

#### 通信量

前向 和 反向 各需一次 All_Reduce	

## Embedding Block（输出层）

代码和理论的实现方式或许不一样

#### 代码

![crossentropy](/Users/guokunhao/笔记/parallelism/megatron1/crossentropy.png)

![crossenyropy2](/Users/guokunhao/笔记/parallelism/megatron1/crossenyropy2.jpeg)

## 随机种子设置

**一般在TP/PP组内，设定不同的随机种子。而在DP组内，设定相同的随机种子** 
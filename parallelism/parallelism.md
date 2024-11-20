# Pipeline parallelism

## Contributions

- 减少总训练时间中的通信占比（解决了 DP 中的通信消耗瓶颈问题）

**与 DP 相比的优势** 

1、 通信量更少（只需要通信某几层的输出）

2、 通信和计算重叠（激活值和梯度的异步通信）

**怎么做到**

- 通信和计算是同时进行的
- 自动划分 stage（通过profile阶段的分析，在最大限度减少通信的同时，平衡不同stage之间的计算负载）
- 在一些stage上使用DP，为了平衡计算负载
- 交替进行前向计算和反向计算（不会一直做前向，使模型收敛）
- 交替进行前向和反向，需要处理前向和反向的参数版本

**数据并行** 

1、 通信量与模型大小成正比，模型变得越来越大，训练的大部分时间都用来通信

2、 GPU迭代更新，计算变得更快，使计算通信比变低

**模型并行** 

1、 不能有效切割模型（有强化学习算法去做，但是费时间费资源）

2、 一次只能做一部分计算（去做流水线，一是比较难，二是有前向和反向计算有参数版本不匹配问题）

## Method

![pipeldream](/Users/guokunhao/笔记/parallelism/pipeline/pipeldream.png)

有三个问题需要解决：

1、 根据GPU数，怎么有效的将模型划分为多个stage

2、 在保证有效学习的情况下，怎么调度计算

3、 在异步通信的情况下，怎么做到有效学习

### 划分 stage

**需要考虑两点** 

1. 每个stage大概有相同的工作量
2. stage之间的通信量尽可能的少

**method** 

1. 在一个机器上profile模型
2. 运行partitioning algorithm去切割模型（怎么切割模型，每个stage的DP数）

#### profile

​	需要记录每一层的三个值：

- T~l~ ：这一层前向传播和反向传播的总计算时间
- a~l~ ：这一层 输出激活 和 输入梯度 的大小
- w~l~ ：这一层的参数量

使用 a~l~ 来计算 C~l~  ：层 l 和 层 l+1通信所需要的时间

W~l~^m^  ：使用m台机器进行DP，并使用分布式参数服务器时，同步层l 的参数所需要的时间（使用w~l~ 估计）

#### Partitioning Algorithm

需要确定三件事：

1. 怎么划分stage
2. 每个stage的DP数
3. 确定使吞吐量最大的minibatch数

最小化总体运行时间，等价于最小化最慢的stage的时间，有最优子问题属性，用动态规划求解

![t_ij](/Users/guokunhao/笔记/parallelism/pipeline/t_ij.png)

![A_jm](/Users/guokunhao/笔记/parallelism/pipeline/A_jm.png)

子问题个数为O(NM)，每个子问题的复杂度为O(NM)，所以总的时间复杂度为O(N^2^M^2^)

让流水线处于满载状态的minibatch数：[ (#machines) / (#machines in the input stage)] = NUM_OPT_ACTIVE_MINIBATCHRS(NOAM)

### 计算调度

​	做太多前向计算，而反向传播较少，会阻碍模型学习，收敛时间变长

​	做一个batch的前向，就开始计算反向，会使机器处于空转状态

pi pedream的调度方案：

​	输入NOAM个minibatch，一旦处于稳定状态，每个stage交替执行前向计算和反向计算。处于稳定状态后，没有stage处于空转状态。（one-forward-one-backward  1F1B）

​	<u>对于使用了DP的stage，使用 确定性轮询负载均衡方案 ，把上一stage的输出，均衡负载到该stage上。（minibatchID mod stageReplicaID）可以确保machine对相同的minibatch做前向和反向。</u>

**代价** ：使用了异步通信，这就带来了有效学习的问题（会遇到前向传播和反向传播所用的参数版本不匹配问题）

​	会带来 **负载不均衡** 问题，越靠前的stage，需要保存越多的激活值

### 有效学习

**直接使用1F1B会有两个问题** ：

1. 同一stage的同一minibatch，前向和反向会使用不同版本的参数
2. 不同stage的同一minibatch，会使用不同的参数版本

​	不利于模型收敛

**weight stashing** 

做前向计算时，每个minibatch会用该stage最新版本的参数，并把参数作为中间值进行保存。

做反向计算时，取出对应版本的参数去计算梯度。

只能保证在一个stage内，前向计算和反向计算使用相同版本的参数，但是不同stage的参数版本还是不一致的。

![weight stashing](/Users/guokunhao/笔记/parallelism/pipeline/weight stashing.png)

**vertical sync** 

来解决各stage所用参数版本不一致问题，每个stage都用input stage 中最新的参数版本，相应的信息与激活值和梯度一起传递。

![vertical sync](/Users/guokunhao/笔记/parallelism/pipeline/vertical sync.png)

**weight stashing 是比较重要的，vertical sync的作用可以忽略，所以weight stashing是默认设置**

<u>**代价** ：需要多个minibatch才能进行一次有效学习，会使收敛变慢</u>

### GPU内存处理

根据每层的参数，计算出 输入，权重，中间值所需的内存大小（每个stage是不一样的）。在训练开始时，分配相应的内存大小，并在之后的过程中重复利用这些内存。

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

# ZeRO

## Contirbution

通过消除DP和MP训练过程中的内存冗余，去优化内存，可以做到：

1. 提升训练速度（获得super-linear的速度提升：每个设备的计算速度不变或提升，总的运算速度与设备数量成正比）：可以使用更大的batch size，提升训练速度
2. 可以训练更大的模型（与设备数量成比例提升）

很容易去使用，不需要去重构模型

## Related Work

### Parallelism

1. DP：会在每个进程复制模型参数，所以有糟糕的memory efficiency，但是有较好的computer/communication effiiency 
2. MP：计算粒度较低，同时需要大量communication，所以有较好的memory efficiency，但是有糟糕的computer/communication effiiency 
3. PP：由于横向切割，共享参数不容易实现；由于micro-batch，batch-normalization不容易实现。

​		对于GPipe，需要正比于pipeline partitions数的batch去提高效率，所以设备数不能太多，因为太大的batch size会影响收敛；同时还需要较大的内存去保存activation。

​		对于pipedreaam，需要保存stale parameter去有效更新参数，所以有更低的memory efficiency；同时，不能有效的提升batch size。

​		除此之外，pp还不容易应用。

### Non-parallelism

**Reducing Activation memory**

Compression、activation checkpointing、live analysis 

**CPU Offload**

大部分时间被用来传输数据

**Memory Efficient Optimizer**

通过保存模型参数和梯度的粗粒度（coarser-grained）统计数据，去减少自适应优化器的内存消耗，但对模型收敛有潜在的影响。（ZeRO不会改变优化器的状态）

## Memory Consumption

### Model State

**optimizer state、gradient、parameter**

### Residual State

**activations、temporary buffers、memory fragments**

## ZeRO-DP（ZeRO-powered DP）

**model state不会在训练的所有时间都被用到（现有的方法会在整个训练过程中都保存model state）** 

![ZeRO-DP](/Users/guokunhao/笔记/parallelism/ZeRo/ZeRO-DP.png)

**基于一下3个关键想法**

1. DP比MP有更好的scaling efficiency。MP有较低的计算密度，同时还有较高的通信量。
2. DP由于在每个进程都复制model states，所以有糟糕的memory efficiency。MP则相反。
3. DP和MP会在整个训练过程中都保存model states。

### P~os~  (Optimizer State Partitioning)

​	对于 N~d~ 的数据并行度，把optimizer states分成 N~d~ 个相等的部分，每个进程只更新它所对应的那部分optimizer state，然后只更新对应的parameter，在每个training step结束时，应用all_gather得到完全更新的参数。

**通信量分析**

​	每个进程都会算出所有参数的梯度，可以应用scatter-reduce得到对应部分梯度，去更新对应部分的参数，最后应用all_gather得到完全更新的参数。

​	scatter-reduce的通信量是$\Phi$，all_gather的通信量是$\Phi$，所以总通信量是2$\Phi$。

### P~os+g~  (Add Gradient Partitioning)

​	在反向传播过程中，会得到每层所有参数的梯度，但只reduce该进程要更新的那部分参数的梯度，在reduce后，就不再需要其余部分梯度，因此可以将内存释放掉。

​	在reduce对应部分的梯度时，做的是reduce-scatter操作，对应不同参数的梯度被reduce到不同的进程。在执行做个操作时，采用bucketization strategy，对对应部分的梯度进行bucketize，bucket一满，就进行reduce，通过这样可以重叠通信与计算。（应该是这样）

**通信量分析**

​	每个进程都会算出所有参数的梯度，可以应用scatter-reduce得到对应部分梯度，去更新对应部分的参数，最后应用all_gather得到完全更新的参数。

​	scatter-reduce的通信量是$\Phi$，all_gather的通信量是$\Phi$，所以总通信量是2$\Phi$。

### P~os+g+p~  (Add Parameter Partitioning)

​	每个进程只存储对应部分的参数，在前向传播和反向传播需要计算时，通过broadcast将对应的参数发送给所有的进程。

**通信量分析**

​	在前向传播期间，每个进程需要从其他进程收到其余部分的参数，这个操作可以被流水线执行。在执行这部分参数所对应的模型的前向传播之前，存储这部分参数的进程将该参数broadcast到所有的进程，执行完这部分的前向传播后，这部分参数就被丢掉来节省内存。这部分操作总的通信量为：$\Phi * N_d / N_d = \Phi$

​	所以总的通信量为3$\Phi$。

## ZeRO-R

**ZeRO-R来优化Residual State** 

### P~a~ ：Partitioned Activation Checkpointing

**两个关键想法**

1. MP虽然分区了model states，但是需要重复activation（每个进程都需要相同的activation）
2. 对于很大的模型，算数强度是很大的，并且与隐藏层维度成线性关系，所以就有可能忽略掉activation checkpoints的数据移动（甚至有时候可以讲activation加载到CPU）。

​	一旦一个模型的一层的前向传播计算完，input activation就在分区在所有的MP进程中，只有在这个activation被再一次用来计算时（在反向传播进行重计算时），使用 all_gather 把activation再次形式化为重复的形式。通常与activation checkpointing一起使用。

​	对于非常非常大的模型，并且设备内存是有限的，可以把partitoned activation加载到cpu中，这个操作是P~a+cpu~ 。

**通信量分析**

​	在MP中，每个transformer block，前向传播需要两次all_reduce操作，反向传播的重计算需要两次all_reduce操作，反向传播需要两次all_reduce操作，所以总的通信量是12 * batch_size * seq_length * hidden_dim，all_reduce操作的通信量是 2 * message_size。

​	在ZeRO-R分区了activation checkpoint后，在反向传播进行重计算时，每个activation checkpoint需要额外执行一次all_gather操作，通常在每个transformer block会有input activation checkpoint，all_gather的通信量为meaasge_size = batch * seq_length * hidden_dim。额外增加的通信量少于MP原本通信量的10%。

​	MP与DP一起使用时，通过使用P~a~ 可以使DP的通信量减少一个数量级，但是需要增加10%的MP通信量。P~a~可以降低MP degree倍内存，这就可以成比例的提高batch size。DP的通信量与batch size成反比。所以P~a~可以导致batch size增加一个数量级，所以DP通信量就减少一个数量级。

​	如果说尽管使用了P~a~，但还是因为batch size太小，DP的通信成为瓶颈，只要CPU数据传输开销小于DP通信开销，此时可以使用P~a+cpu~通过增加batch size，来提高效率。

### C~B~ ：Constant Size Buffers

​	对于一些操作，其计算效率高度依赖于输入的大小，输入越大，效率越高。（如all_reduce）一些高性能计算库，为了计算效率，会把所有参数放在一个缓冲区中，这会消耗极大的内存。

​	在模型很大时，我们使用性能高效的恒定大小的融合缓冲区，使得缓冲区大小不会随着模型变大而变大，同时保持较高的计算效率。

### M~D~ ：Memory Defragmentation

​	内存碎片化是短生命周期内存对象与长生命周期内存对象交错的结果（checkpointed activation是长，discarded activation是短，parameter gradient是长，activation gradient是短）。会带来两个问题：1、OOM：因为没有连续的内存；2、memory allocator 花费大量时间来搜索连续的内存以满足内存请求。

​	通过为激活检查点和梯度预先分配连续的内存块，并在生成它们时将它们复制到预分配的内存中，动态执行内存碎片整理。这可以：1、增加可用的内存。2、减少memory allocator查找空闲连续内存所需的时间。

## ZeRO 和 MP

​	1、在减少每个设备所被占用的内存方面，ZeRO-DP和MP一样高效，甚至当MP不能均衡的切割模型时，ZeRO-DP会更加高效。2、ZeRO-DP 有更好的scaling efficiency。3、ZeRO-DP和DP一样，很容易的应用。

但是，还是有两种情况需要应用MP：

1. 对于大模型训练，与ZeRO-R一起使用，MP可以减少activation memory
2. 当仅使用DP导致有太大的累积batch size时（所有设备都用作DP），会导致模型不能收敛的很好（太大的batch size会降低收敛速度）。在这种情况下，可以使MP和ZeRO一起使用，得到一个好的batch size。

# Megatron 2

**高效地训练大模型的挑战**

1. 不可能把一个大模型放在一个多GPU服务器上
2. 需要的计算操作数会导致不切实际的训练时间

**在GPU集群上高效训练模型需要**

1. 高效内核实现：使大多数计算成为compute-bound而不是memory-bound
2. 设备间计算图的智能切割：降低通信量
3. 降低设备空转时间
4. 特定领域的通信优化
5. 优秀的硬件

## Contribution

1. PTD-P（inter-node pipeline par- allelism, intra-node tensor parallelism, and data parallelism）怎样融合起来去达到更高的累积吞吐量（给定batch size，并且在严格的optimizer semantics）
2. 与各种并行度相关的各种权衡
3. 各种并行之间如何相互交互
4. 提出一种交错流水线调度，可以将吞吐量提高10%，内存占用与现有方法相当

**分布式训练的指导性原则**

1. 不同形式的并行化会有不同的影响：并行化策略会影响通信量、内核执行的计算效率、流水线刷新带来的空转时间。在单个服务器内，TP比较高效；PP适用于比较大的模型。
2. PP的不同调度会影响通信量、流水线bubble大小和存储激活的内存大小。
3. 超参数会影响内存占用、内核执行的计算效率和流水线bubble大小。
4. 分布式训练是communication- intensive（通讯密集型）。

这篇文章，没有搜索并行化策略空间，而是使用启发式，也很有效。

## Related Work

### **data parallelism**

**DP有好的scale efficiency，但有两个限制（DP的batch size是固定的）：**

1. 设备超过一定数，每个设备的batch size太小，会降低GPU计算时的利用率，同时会增加通信消耗
2. 最多的设备数量是batch size，限制了训练时可以使用的设备数量

### **tensor parallelism**

**使用TP，大模型需要的多个多GPU服务器间切割模型，会带来两个问题：**

1. 在服务器间使用all_reduce通信，会有很慢的通信速度
2. 在TP并行度比较高时，每个设备都做小的矩阵乘法，会降低GPU的利用率

### **pipeline parallelism**

​	不同的层分配和调度策略（前向传播和反向传播如何调度）会有不同的性能权衡。不管是哪种调度，为了保持严格的optimizer semantics，优化步需要在每个设备间同步，这个就会导致在每个batch结束时进行pipeline flush，这会降低计算吞吐量。micro batch数和pipeline size的比例越大，pipeline flush的时间越少，所以通常会有一个较大的micro batch数。

​	一个batch会被分成多个小的micro batch，在多个micro batch间流水线执行前向传播和反向传播，为了保持严格的optimizer semantics，就需要周期性的流水线刷新。在每个batch的开始和结束，设备处于空转状态，把这个空转时间成为pipeline bubble。一些异步和bounded- staleness方法没有流水线刷新，但是弱化了权重更新semantic，这不在本文讨论范围内。

#### Schedule

t~pb~ : pipeline bubble	m : the number of microbatches	p : the number of pipeline stages

t~id~ : the ideal(perfect) time per iteration	t~f~   t~b~  : the time to execute a single microbatch’s forward and backwardpass

**Default Schedule**

![GPipe](/Users/guokunhao/笔记/parallelism/megatron2/GPipe.png)

GPipe Schedule

$t_{pb} = (p - 1) * (t_f + t_b)$

$t_{id} = m * (t_f + t_b)$

Bubble  time fraction(pipeline bubble size)$ = \frac{t_{pb}}{t_{id}} = \frac{p - 1}{m}$

为了使bubble time变小，m需要变大，这会导致更高的内存占用，需要保存m个micro batch的中间激活。

![pipedream](/Users/guokunhao/笔记/parallelism/megatron2/pipedream.png)

PipeDream-Flush Schedule

​	首先进入warm-up phase（每个进程执行不同数量的前向传播），这种调度方式会将正在执行（反向传播还没有执行，需要保存相关的激活值的microbatches）的microbatches数限制为流水线深度，而不是一个batch中的microbatches数。

​	在warm-up phase后，进程进入steady state，进行1F1B。最后，执行正在运行的microbatches的反向传播。

​	bubble time大小与GPipe调度一样，未完成的正向传递的数量最多是 PipeDream-Flush schedule的流水线阶段数，所以需要保存激活值的microbatches数最多是p，更加memory-efficient。

**Schedule with Interleaved Stages**

![Schedule with Interleaved Stages](/Users/guokunhao/笔记/parallelism/megatron2/Schedule with Interleaved Stages.png)

​	思路：为了减小pipeline bubble的大小，每个设备可以执行多个子层集合（模型块）的计算，而不是单一的连续子层集合。这样，流水线中的每个设备就被分配了多个stage，所以每个stage有更少的计算。

​	该调度仍采用1F1B，一个batch中的micro batches数应该是流水线并行度的整数倍。

对于相同的batch size，流水线刷新更快：每个设备有 v 个stage

对于一个microbatch，前向传播和反向传播的时间分别为：$\frac {t_f}{v}$  $\frac {t_b}{v}$

$t_{pb} = \frac{(p - 1) * (t_f + t_b)}{v}$

$t_{id} = m * (t_f + t_b)$

Bubble  time fraction(pipeline bubble size)$ = \frac{t_{pb}}{t_{id}} = \frac1 v * \frac{p - 1}{m}$

Bubble time减少至原来的 v 倍，但会带来额外 v 倍的通信量。

### Weak scaling

使用weak scaling setup，去评估默认非交错流水线并行调度的scale性能。

随着增加pipeline stage数，通过成比例的增加模型中的层数，来增大模型大小。对于所有的设置都使用大小为8的TP。

![weak scaling](/Users/guokunhao/笔记/parallelism/megatron2/weak scaling.png)

由结果可知，对于更大的batch size，scale效果更好。因为随着stage增大，pipeline bubble也变大，但对于更大的batch size，pipeline bubble会分摊在更多的microbatch中。

### Interleaved versus Non-Interleaved Schedule

![Interleaved versus Non-Interleaved Schedule](/Users/guokunhao/笔记/parallelism/megatron2/Interleaved versus Non-Interleaved Schedule.png)

带有scatter/gather通信优化的 interleaved schedule 比 non-interleaved schedule 的效果更好，但它们之间的差别会随着batch size的增大而减小，有两个原因：

1. 随着batch size的增大，default schedule的pipeline bubble 减小
2. pipeline中的点对点通信量正比于batch size，由于interleaved schedule 每个microbatch有更多的通信量，所以default schedule随着通信量的增加会逐渐缩小差距

如果没有scatter/gather通信优化，在更大的batch size时，non- interleaved schedule 比 interleaved schedule 的效果更好。

## Performance Analysis of Parallelization Configurations

不同并行策略之间的权衡

### Notation

(p, t, d) : 并行化维度	n : GPU数	B : global batch size	b : microbatch size

$m = \frac1b * \frac Bd$ : 每个流水线中一个batch的microbatch数

### Tensor and Pipeline Model Parallelism

假设 d = 1，所以  $pipeline\,bubble\,size=\frac{p - 1}{m} = \frac{n/t-1}{m}$

对于固定的B, b, d，则m也固定。随着t增大，pipeline bubble变小。

根据p和t的不同，通信量也会有所不同。PP是成本较低的点对点通信，而TP需要all_reduce通信操作。在PP中，对于每个microbatch，在每对连续设备的前向传播和反向传播需要的通信量是 $bsh$ 。在TP中，对于每个microbatch，在每个设备的每一层中，大小为$bsh$ 的数据需要在前向传播和反向传播中各all_reduce两次，所以总的通信量为  $$l^{stage} * （8bsh(\frac{t-1}{t}))$$ 。

![Tensor and Pipeline Model Parallelism](/Users/guokunhao/笔记/parallelism/megatron2/Tensor and Pipeline Model Parallelism.png)

TP最好在一个服务器中。PP中较多时间花费在pipeline bubble中，所以应该限制PP的stage数，以便micro batch数是stage数的合理的倍数。

**总结1**：在考虑不同形式的MP时，当使用g个GPU的服务器时，TP度应该为g，PP可以跨服务器扩展到更大的模型。

### Data and Pipeline Model Parallelism

设 t = 1，每个流水线的microbatches数是  $m = \frac{B}{d*b} = \frac{b^`}{d}$ ，

其中$$b^` = \frac Bb$$，$p = \frac nd$

$pipeline\,bubble\,size = \frac{p-1}{m} = \frac{n/d-1}{b^`/d} = \frac{n-d}{b^`}$

所以 d 越大，bubble size越小。ring all_reduce的通信时间是$\frac{(d-1)V}{d\beta}$ （V是通信量，$\beta$ 是带宽），所以通信时间随d的缩放率是$\frac {d-1}{d}$ 。所以尽管随着d的增大，通信时间会增大，但不会随着d剧烈增大，总的吞吐量最终是会变大的。

对于给定的并行化配置，提高batch size大小B，$b^`$会增大，导致m变大，因此bubble size会减小。同时，数据并行所需的all_reduce操作会变少，吞吐量会进一步增大。

![Data and Pipeline Model Parallelism](/Users/guokunhao/笔记/parallelism/megatron2/Data and Pipeline Model Parallelism.png)

随着PP的增大，吞吐量会减小（随着DP的增大，吞吐量会增大）。所以PP应该主要用于支持单个设备放不下的模型，而DP用于扩大训练规模。

### Data and Tensor Model Parallelism

跨服务器执行all_reduce操作成本非常高。TP需要对每个microbatch执行all_reduce操作，而DP只需要对每个batch执行all_reduce操作。除此之外，每个TP进程只执行一个层的计算子集，效率较低。

![Data and Tensor Model Parallelism](/Users/guokunhao/笔记/parallelism/megatron2/Data and Tensor Model Parallelism.png)

batch size越大，DP的数据通信会减少（因为需要通信的batch数减少）。

**总结2**：在使用数据并行和模型并行时，应该使用使模型参数和中间值能够存到GPU内存中的模型并行数M = t * p，然后用数据并行把模型训练扩展到更多的GPU上。 

### Microbatch Size

给定并行化配置（p，t，d）和batch size大小B的情况下，确定最优的microbatch size大小b。

不论b的大小是什么，DP的通信量是不变的。

函数$t_f(b)$  $t_b(b)$  大小为b的microbatch前向传播和反向传播计算所需的时间，那么计算一个batch所需要的总时间为：$(b^`/d + p - 1)*(t_f(b) + t_b(b))$

b通过改变m来影响bubble size，同时也会影响算数强度。

![Microbatch Size](/Users/guokunhao/笔记/parallelism/megatron2/Microbatch Size.png)

**总结3**：最佳的microbatch大小b，取决于模型吞吐量、内存占用、流水线p、DP度d和batch size大小B。

### Activation Recomputation

activation checkpoints的数量不会影响吞吐量，但会影响内存占用。

A~input~ 是一个层的input size，A~intermediate~ 是一个层的中间激活值大小，一个model stage 有 l 个层，有c个checkpoints，那么总的内存占用是：$c*A_{input} + l/c*A^{intermediate}$，当$c = \sqrt{l*(A^{intermediate}/A^{input})}$ 时，内存占用最小。通常情况下，每1～2个transformer layers 去checkpoint是最优的。

![Activation Recomputation](/Users/guokunhao/笔记/parallelism/megatron2/Activation Recomputation.png)

对于小的batch size，由于反向传播前的激活重计算，会导致一个比较低的吞吐量；但是使用了激活重计算，就可以使用较大的batch size，这可以减小bubble size，从而提升吞吐量。

## Implementation

### Communication Optimizations

可以使用TP和PP，去减少跨节点间通信开销。

每个 DGX A100 都配备了 8 个 InfiniBand （IB） 网卡。但是pipeline中的send和receive都是点对点通信，只能在两个GPU间进行通信，所以在pipeline中就不能有效的利用8个网卡去进行通信。

在不同的TP进程中，每个对应的transformer layer都会有相同的重复的输出，这就会导致多个进程在相邻的两个流水线stage间send和receive相同的tensor。为了消除这些冗余，可以在发送侧把tensor切割成相同的块，然后每个进程使用自己的网卡将对应块发送给下一个stage相应的进程。然后在接受侧使用all_gather去得到完整的tensor，因为TP一般在同一个服务器中，传输数据更快。将这个称为 scatter/gather communication optimization。

通过这个优化，每对连续stage之间的通信量减少为$\frac{bsh}t$

![scatter:gather ](/Users/guokunhao/笔记/parallelism/megatron2/scatter:gather .png)

### Computation Optimizations

对计算图实现三种特定于模型的优化，以此来实现高性能：

1. 改变transformer layer的数据布局，来避免memory- intensive转置操作，并支持使用跨batch矩阵乘法内核（strides batched GEMM kernels）：将数据布局从[b，s，a，h]改为[s，b，a，h]。
2. 为一系列元素操作生成融合内核（bias + GeLU    bias + dropout + add）。
3. 实现两个自定义内核使得能够融合scale、mask、softmax(reduction)操作：一个支持常规masking（在BERT中使用的）；另一个支持implicit causal masking（在类似于GPT的自回归模型中使用）。

## Evaluation

### Comparison to ZeRO-3

由于PTD-P比ZeRO-3有更少的跨节点间通信，所以PTD- P会更优。

# Megatron-LM3	Sequence Parallelism

## Contribution

在减少activation内存占用的同时，不会引入其他额外的开销，从而因为减少activation recomputation，加速模型训练速度。（只适用于transformer结构）

1. sequence  parallelism：与TP一起使用可以大量减少activation内存占用
2. selective activation recomputation：可以减少由 full activation recompilation 带来的90%的计算开销。

## Activation Memory

### Activation Memory Per Transformer Layer

#### tensor parallelism

![tensor parallelism](/Users/guokunhao/笔记/parallelism/megatron3/tensor parallelism.png)

**Attention Block**

1. Layer norm：需要保存输入：2bsh
2. Q、K、V矩阵乘法：需要保存它们的共享输入：2sbh
3. QK^T^ 矩阵乘法：需要分别保存Q和K：2sbh / t、2sbh / t
4. softmax：需要保存softmax的输入：2as^2^b / t（a 是 head_num）
5. softmax dropout：需要保存mask：as^2^b / t
6. attention over values(V)：需要保存dropout的输出和V：2as^2^b / t、2sbh / t
7. linear project：需要保存输入激活值（attention over values的输出）：2sbh / t
8. attention dropout：需要保存mask：sbh

所以attention- block需要保存的大小为：5sbh + 8sbh / t + 5as^2^b / t

**MLP**

1. layer norm：需要保存输入：2sbh
2. 第一个线性层的输入：2sbh
3. GeLU非线性层的输入：8sbh / t
4. 第二个线性层的输入：8sbh / t
5. linear dropout的mask：sbh

所以MLP- block需要保存的大小为：5sbh + 16sbh / t

**对于transformer的一个layer需要保存的激活内存占用为：$sbh(10+\frac{24}{t}+5\frac{as}{ht})$**

#### Sequence Parallelism

​	在TP中，layer-norm和在attention、MLP block之后的dropout都需要完整的数据（没有在TP进程组中切割），这些数据在TP进程组中是重复的，从而导致TP activation内存占用中的10sbh没有在TP进程组中切割。

​	在transformer layer中的non TP 区域，数据的操作在sequence dimensions 是相互独立的，所以我们可以沿着sequence dimensions 去切割数据，从而节约activation内存占用。在sequence dimensions 进行并行后，会额外引入集合通信操作。在前向传播中，在执行 f 之前，需要执行一次all_gather操作，在执行 $\bar f$  之后，需要执行一次scatter操作。

![sequence parallelism](/Users/guokunhao/笔记/parallelism/megatron3/sequence parallelism.png)

**MLP**

![SP-MLP](/Users/guokunhao/笔记/parallelism/megatron3/SP-MLP.png)

​	下标表示设备编号，上标表示按照哪个维度进行切割。layer- norm的输入size是[s，b，h]。

​	对layer-norm的输入在sequence维度进行并行化 $X = [X_1^s,X_2^s]$，layer-norm的输出也将在sequence维度进行并行。对于带GeLU非线性的线性层，需要完整的 Y 作为输入，所以在前向传播阶段 g operator需要做一次all_gather操作。然后对矩阵A和矩阵B分别进行列切割和行切割进行并行（TP）。在进入 dropout 之前W~1~、W~2~ 需要加起来（即在TP中做一次all_reduce），在进入dropout后，数据需要沿着sequence dimensions维度进行切割。所以我们可以把这两个操作和在一起（相加和切割），进行一次reduce-scatter操作。所以在前向传播阶段。$\bar g$  需要进行一次reduce-scatter操作。总的计算过程如下：

![SP-MLP-equation](/Users/guokunhao/笔记/parallelism/megatron3/SP-MLP-equation.png)

$g\,和\,\bar g $  是共轭操作，g 在前向传播中做all_gather操作，在反向传播中做reduce_scatter操作；$\bar g$ 在前向传播中做reduce_scatter操作，在反向传播中做all_gather操作。

​	通过 TP + SP ，反向传播所需要的中间激活都被分割在多个设备上，但第一个线性层所需的输入 Y 仍没有被分割（在transformer layer 中是求Q、K、V时的输入）。为了解决这个问题，我们把 Y 沿着sequence dimensions进行切割，然后进行一次额外的all_gather操作，再通过重叠通信和计算来消除在反向传播时带来的通信消耗。

**Transformer**

​	在transformer block中的操作类似。

**通信量分析**

​	在TP中，一次前向传播和反向传播，需要4次all_reduce；在SP + TP 中，一次前向传播和反向传播，需要4次all_gather，4次reduce_scatter。所以两种通信量相同。

**对于transformer的一个layer需要保存的激活内存占用为：$\frac{sbh}{t}(34+5\frac{as}{h})$**

#### Pipeline Parallelism

​	为了减少pipeline bubble，PP schedule会引入一个stage存储多个激活的问题，从而导致PP不会在每个设备上均匀的存储激活。

​	对于1F1B，为了减少pipeline bubble，第一个stage具有最大的内存压力，其必须存储 p 个micro batch。每个stage包含 L / p 层，所以不论 p 的大小是什么，第一个stage必须存储 p * L / p = L 层的激活值，其所需要存储的激活内存大小是：$\frac{sbhL}{t}(34+5\frac{as}{h})$

### Total Activation Memory

![transformer](/Users/guokunhao/笔记/parallelism/megatron3/transformer.png)

​	除了transformer layer，还有input embedding、最后的layer-norm和output layer的激活内存占用需要计算。

​	position and word embedding没有大量的激活值需要存储，但是embedding layer中的dropout有激活值需要存储，其也沿着sequence dimensions 切割，所以需要的内存大小是：sbhp / t（PP中第一个stage需要p个microbatch）（在dropout前需要做reduce_scatter操作，也就是有$\bar g$）。

​	最后的layer-norm也需要SP，所以需要的内存大小是：2sbh / t 。output layer的projection需要保存其输入，占用的内存大小是：2sbh / t；在cross entropy loss之前需要一次all_gathe（也就是有$g$），最后，cross entropy loss的计算需要存储最后以32位浮点数格式存储的logits，其占用的内存大小是：4bsv / t（TP的原因）。所以占用的内存大小是：$\frac{4sbh}{t}(1+\frac vh)$

​	**总的内存占用为：$\frac{sbhL}{t}(\frac pL + \delta_{p=1}\frac 4L(1+\frac vh))$**  （其中$当p=1时，\delta_{p=1}=1，否则为0$）

​	这相对于transformer layer的内存占用是可以忽略。

## Selective Activation Recomputation

对于 L 个transformer layer，激活内存占用大小为： $\frac{sbhL}{t}(34+5\frac{as}{h})$

​	checkpoint所有的transformer layer，会节省大量的内存，但会带来30%-40%的计算时间开销。为了平衡内存节省和计算开销，理想的做法是对于给定的模型，在给定的受限制的设备上尽可能的checkpoint activation。SP+TP可以节省一些显存，但最优的模型并行配制一般需要保存activation和recomputation activation一起使用。选择保存激活和重计算激活的数量的一个简单方法是，只checkpoint一些transformer layer，而保存其他layer的所有激活。但是这个方法不能很好的扩展到大模型上。例如，在训练 MT-NLG 时，每个设备只有三个层，这限制了可以平衡内存与计算的粒度。

​	不是所有的激活值都需要相同数量的操作去重计算（有的激活值需要的运算数较少，有的激活值需要的运算数较多），所以我们可以在选择哪些激活值进行存储，哪些激活值进行重计算上进行一些创新。所以，我们可以这样做：不去checkpoint和重计算一个transformer layer的所有值，而是checkpoint和重计算一个transformer layer的一部分，这一部分占用了大量显存，但重计算的计算代价较低。

​	上述公式中的 as / h ，这是为了提升模型宽度所做的attention操作引起的，所做的操作包括softmax、softmax dropout，这些操作通常需要较大的输入大小，所以会有较大的内存占用，但是所需要的计算代价较小。同时，对于大模型，as / h > 34，所以我们对这一部分进行checkpoint和重计算，可以减少一半多的激活内存，但所需要的重计算代价较小。

​	使用这种 selective activation recomputation，所需要的内存占用减少为  $34\frac{sbhL}{t}$ 。但在反向传播中进行重计算的操作数是：attention matrix计算（2bs^2^h）和attention over values（2bs^2^h）。

## Evaluations

实验中的模型配置如下：

![model configuration](/Users/guokunhao/笔记/parallelism/megatron3/model configuration.png)

### Memory Usage

每个transformer layer，在不同的技术下所需要的activation内存总结如下：

![activation memory](/Users/guokunhao/笔记/parallelism/megatron3/activation memory.png)

每种技术都可以将所需的内存降低至一半左右，两种技术融合起来可以将所需的内存减少5倍，降低至原来的20%左右，这只是full activation recomputation的2倍左右。

### Execution Time per Layer	

对于22B模型，一个transformer layer的前向传播和反向传播所需的执行时间如下：

![execution time](/Users/guokunhao/笔记/parallelism/megatron3/execution time.png)

​	前两行表明，SP可以提高训练速度，缩短计算时间，这主要是由于layer-norm和dropout只在 1 / t 的数据上进行计算。这是SP的主要优势的额外的好处（主要优势是，减少activation内存占用）。同时，通过实验还发现，reduce-scatter和all_gather分开执行，比一起执行，更慢，这就减少了SP对性能的提升。

​	后两行表明，selective recompute可以减少大部分反向传播过程中的重计算，selective recompute会增加11%左右的重计算，而full activation recompute会增加64%的重计算。（这里论文中还有没看懂的地方）

### End-to-End Iteration Time

![end-to-end time](/Users/guokunhao/笔记/parallelism/megatron3/end-to-end time.png)

model FLOPs：不论实现和硬件限制是什么，做一次前向传播和反向传播所需的浮点运算次数。是独立于实现和硬件的，只依赖于模型。

hardware FLOPs：在一个设备上，每个iteration，实际所需的浮点运算次数。所以一个实现需要activation recompute，那么其hardware FLOPs通常会大于model FLOPs。

model FLOPs per second 和 hardware FLOPs per second ：model FLOPs 和 hardware FLOPs 分别除以iteration的时间。

model FLOPs utilization(MFU) 和 hardware FLOPs utilization(HFU) ：model FLOPs per second 和 hardware FLOPs per second 分别除以 加速器（GPU）理论上 每秒浮点运算次数的峰值。


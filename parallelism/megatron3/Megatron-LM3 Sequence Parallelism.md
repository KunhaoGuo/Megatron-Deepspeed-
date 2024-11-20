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
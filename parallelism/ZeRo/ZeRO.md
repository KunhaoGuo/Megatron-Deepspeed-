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
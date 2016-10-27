# cnn_text_classification
  此工作完全基于Denny的教程,本人愚昧,详细记录,以备后用,或微助游客(代码中加入自己的详细注释).
	
  本文主要内容:工作的详细步骤说明以及TensorFlow的相关知识说明.下面是要实现的模型:
  ![model](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/11/Screen-Shot-2015-11-06-at-8.03.47-AM.png)

一:数据预处理

  我们使用影评的[数据集](http://www.cs.cornell.edu/people/pabo/movie-review-data/)
  
  用data_helper脚本来清洗数据,加载句子和标签,还有生成随机的batch.

二:网络结构说明

再放一张图帮助理解:![model2](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/11/Screen-Shot-2015-11-06-at-12.05.40-PM.png)

1.我们来自定义一个类TextCNN来承载网络的各种参数和配置,并在__init__()函数中初始化来生成Graph.

定义网络的输入:input_x,input_y,dropout_keep_prob,在训练和测试的时候需要feed这些placeholder.

@@ TensorFlow也提供这样的机制:先创建特定数据类型的占位符(placeholder)，之后再进行数据的填充. 但是placeholder节点被声明的时候是未被初始化的，也不包含数据，所以在train过程时，需要为这些placeholder节点提供数据,所以在train或test时候通过feed_dict供给数据.

2.定义 EMBEDDING LAYER, output:[None, sequence_length, embedding_size, 1]

这也是网络的第一层,词向量层.是为了把高维的词汇表中词的索引转化成低维的向量表征(vector representation).

如结构图所示，输入层是句子中的词语对应的word vector依次（从上到下）排列的矩阵，假设句子有 n 个词，vector的维数为 k ，那么这个矩阵就是 n×k 的。

@@ TensorFlow的conv2d()函数的输入所需是:tensor of shape [batch, in_height, in_width, in_channels] ,4-D.
tf.random_uniform(shape, minval,maxval),输出一个随机值(范围是[minval, maxval))来填充的的指定shape的tensor.
tf.nn.embedding_lookup(self.W,input_x),这一步之后产生一个3-D的tensor:[None, sequence_length, embedding_size],
然后用tf.expand_dims()函数把3-DK扩展成4-D:[None, sequence_length, embedding_size, 1],对应与con2d()所以的tensor.

3.定义卷积和池化层, outut:[batch_size, 1, 1, num_filters]

模型使用不同size的卷积核(filter),对于不同的filter要分别处理,他们生成tensor的shape是不一样,最后需要把所有结果合并起来,装在pooled_outputs[]中.
首先定义W:filter matrix 和b:bias matrix,

W: a filter tensor of shape[filter_height,filter_width,in_channels,out_channels],[词个数,embedding_size,输入通道,输出通道]
b: a bias tensor of shape [out_channels],他跟卷积层的输出通道数对应.

把第一层得到的词向量和W放入卷积函数con2v(),这里执行narrow convolution,每一次卷积之后得到的tensor的shape是:[1,sequence_length-filter_size+1,embedding_size,1],然后对卷积的输出做一次非线性的激活函数,得到的tensor的shape是:[1,sequence_length-filter_size+1,1,1].

再对这个结果执行max_pool(value, ksize, strides, padding, data_format="NHWC", name=None),在这里设置ksize为[1, sequence_length - filter_size + 1, 1, 1],而input的value的shape为:[[1,sequence_length-filter_size+1,1,1]],且stride为[1,1,1,1].所以,在使用某一size的filter中执行max_pool()后,所得到的结果为: [batch_size, 1, 1, num_filters],结果是一维的,就是每个输出通道有一个值,所以第四维是num_filter,这第四维也就对应求得的feature.

当得到所有filter中执行max_pool后的输出tensor后 ,把他们都合并起来,最后得到的结果:[batch_size, num_filters_total].

为了后面的dropout做准备,还可以flatten这个结果,最后就成一个一维的数组,每个元素就是每个通道的最终输出值.[f1,f2,f3,,,,,]

4.dropout层

为了有效防止过拟合,在max_pool之后加一层dropout,这是一个全链接层.

5.Final (unnormalized) scores and predictions

使用上层得到的结果,通过矩阵运算(xw_plus_b),可以求得最终的score,并且根据最高score来预测它所属的类别predictions.在这一步,我们也可以使用 softmax 函数把原始得分(raw score)转化为标准化概率.

6.LOSS AND ACCURACY

使用上面的score可以进一步定义网络的loss和accuracy,loss值用来衡量网络的错误,我们的目的是让这个值尽可能减小.在分类问题中常用的损失函数是[ cross-entropy loss](http://cs231n.github.io/linear-classify/#softmax).用它来计算每个类的cross-entropy loss,给我们score和对应的lebel.然后计算loss的平均值来作为最终的loss值.

最后定义accuracy,是训练和测试时的一个非常有用的指标.


  

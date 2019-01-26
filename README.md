# Easy_Rnn_Attention
RNN+attention 使用预训练词向量 中文文本分类

# 数据集：
本实验是使用THUCNews的一个子集进行训练与测试，数据集请自行到THUCTC：一个高效的中文文本分类工具包下载，请遵循数据提供方的开源协议;

文本类别涉及10个类别：categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']，每个分类6500条数据；

cnews.train.txt: 训练集(5000*10)

cnews.val.txt: 验证集(500*10)

cnews.test.txt: 测试集(1000*10)

训练所用的数据，以及训练好的词向量可以下载：链接: https://pan.baidu.com/s/1daGvDO4UBE5NVrcLaCGeqA 提取码: 9x3i 

# 1.利用Rnn+attention进行文本分类
## 模型参数
parameters.py

## 预处理
预训练词向量进行embedding

对句子分词，去标点符号

去停用词

文字转数字

padding等

标准做法：最后，根据动态RNN模型的特点，对每个batch中的句子，根据batch中最长句子的max_length,对其他句子进行padding，也就是在后面补0，同时计算各句子的真实长度，存入列表。为啥要计算真实长度？因为有用啊！！！因为给动态RNN输入真实的句子长度，它就知道超过句子真实长度后面padding的0是无用信息了。

真实做法：由于句子的长度实在是太长了，保持句子的原长，小霸王电脑运行不出来，所有对句子进行截断处理，将所有句子长度通过截断与padding保持为250.然后计算句子的真实长度。

程序在data_processing.py

## 运行步骤
Training.py 

由于小霸王运行非常吃力，因此只进行了3次迭代。

![train and test result](https://github.com/NLPxiaoxu/Easy_Rnn_Attention/blob/master/image/train.jpeg)

predict.py 模型用来对验证文本进行预测

![evalutaing result](https://github.com/NLPxiaoxu/Easy_Rnn_Attention/blob/master/image/eva.jpeg)

验证结果表明，5000条文本准确率达96.5%，取前10条语句的测试结果与原标签对比。

# 参考

https://blog.csdn.net/thriving_fcl/article/details/73381217 

https://github.com/cjymz886/text_rnn_attention

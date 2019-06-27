孪生LSTM网络
====
本项目是基于孪生LSTM网络+曼哈顿距离(Manhattan distance)实现的句对相似度计算<br>
训练集
---

* 数据生成：选用的是数据集是33007条标注过的语料，总共有60类，数据在data/train_lstm_20190303_data.csv ，经过数据分析可知，有17类的样本数小于10，于是将样本数小于10的类删除，得到32922条数据共43类。
因孪生网络的输入是句子对，于是考虑将相同类的样本组成的样本对作为正样本，不同类的样本组成的样本对为负样本。<br>
* 正样本对：采用随机采样的思想，对每一类中的各个样本，随机选取10条相同类的样本作为其正样本对，最后得到的正样本数为329220个。<br>
* 负样本对：根据样本的类型选取负样本对，同样，采用随机采样的思想，对每一组中的各个样本，随机选取10条不同类的样本作为其负样本对，
最后共产生负样本对的个数为216299个，正负样本比例接近1.5:1。正负样本数据对保存在siamese_data文件夹下。<br>
* 最终的训练样本保存在 data/sent_pair_new.csv<br>
* 将上诉生成数据集去重，得到392302条数据，其中正样本包含255967条，负样本包含136335条，正负样本比例接近1.88:1，训练样本保存在 data/sent_pair_dup1.csv<br>
* 将数据生成的方式改为只是在负样本生成上改为随机选取20条不同类的样本作为其负样本对，最后共产生负样本对的个数为432343个，正负样本比例接近1:1.3。
  数据对保存在 siamese_data523文件夹下。将数据对整合后最终的训练样本保存在 data/sent_pair_new.csv。<br>
* 将上一步生成的数据集去重，最后共产生负样本对的个数为260072条，正样本个数为255966条，正负样本比例接近1:1.数据保存在 data/sent_pair_dup524.csv。<br>

测试集
---

* 从3月18-3月24日的benrenshoucui数据，按in_node等比例抽样，共3004条数据。对这3004条数据进行数据扩充，共47361条。数据保存在 data/test.csv<br>

![](https://cloud.githubusercontent.com/assets/9861437/20479493/6ea8ad12-b004-11e6-89e4-53d4d354d32e.png)




Reference
----
(1)[Siamese Recurrent Architectures for Learning Sentence](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12195)<br>
(2)[Learning Text Similarity with Siamese Recurrent Networks](https://www.researchgate.net/publication/304834009_Learning_Text_Similarity_with_Siamese_Recurrent_Networks)



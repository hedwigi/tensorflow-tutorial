# 推荐系统小实验
此模型将"好友推荐"任务看作多标签分类问题。
与传统多标签分类问题的不同之处在于，测试集的uid和fid必须都在训练集中出现过，也就是符合推荐问题的设定。

因为希望充分利用训练数据，所以对于每个用户的表现不仅包括user profile，还会包括训练集中fid的表现。
而预测目标，不仅包括valid数据中的fid，还包括训练集的fid。

测试时，给定一组已知的uid，预测的结果主要包含所有在train和val中见过的该uid的fid（分值理论上接近1）；将这部分fid排除掉之后的top-k结果，即最终的推荐结果

训练数据：U-F matrix train
验证数据：U-F matrix valid

valid中包含所有train中的uid和fid

## 输入表现

输入为uid

user_vocab = 8 

为每一个social.train的user进行表现

- For each user, one-hot representation for every feature(with different vocab size)
```
gender:
gender_vocab = 2
u1: [0,1]
u2: [1,0]

city:
city_vocab = ?
u1: [0,0,0,1,...,0]
u2: [0,1,......,0]

country:
...

tag:
...

tweet:
u1: [0,0,1,0,1,1,1,0,0,1,..]
u2: [1,0,1,1,0,0,0,1,0,1,..]

friend train:
u1: [0,0,0,1,1,1,0,1,0,0,1,1]
u2: [0,1,1,0,0,1,0,1,0,0,1,0]
```

- Initialize embeddings for every feature

- For each user, emb for every feature
```
gender:
emb_size = 2
u1: gd0=0|gd1
u2: gd0|gd1=0

city:
emd_size = ?
u1: ct0=0|ct1=0|ct2=0|ct3|ct4=0|...
u2: ct0=0|ct1|ct2=0|ct3=0|...

country:
...

tag:
...

tweet:
u1: w0=0|w1=0|w2|w3=0|w4|.....
u2: w0|w1=0|w2|w3|w4=0|.....

friend:
u1: ...
u2: ...
```
 
- For each user, concat the embeddings of all feature values
```
gd0|gd1|ct0|ct1..|ctry0|ctry1...|tg0|tg1..|w0|w1...|f0|f1|...
```

- transform to feature map

## 模型
CNN

## 输出

output size = train friend_vocab
multi-hot representing fid in train + val，因为如果输出标签只有valid则很不合理，明明train中的fid也应当是分值很高的

# notes

## 词汇表处理

```python
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import lookup_ops

sess = tf.Session()

# list of string
vocab = ['<PAD>', '<EOS>', '<UNK>', '你', '…', '了', '回家', '米', '?', '咯', ',', '。', '肺', '!', '又', '[怒]', '个', '好', '因为', '说话', '中秋', '所以', '[泪]', '不要', '要']
UNK_id = vocab.index('<UNK>')

# Tensor("Const:0", shape=(25,), dtype=string)
mapping_strings = tf.constant(vocab)

# HashTable, string -> int
vocab_table = lookup_ops.index_table_from_tensor(mapping_strings, default_value=UNK_id)

id_you = vocab_table.lookup(tf.constant('你')).eval(session=sess)
ids_lw = vocab_table.lookup(tf.constant(vocab)).eval(session=sess)

```

## data iterator


每一个data_iterator是一个Data实例


run_one_batch时候，调用iteratior.get_next_batch

## model
tf.global_variables_initializer()
tf.local_variables_initializer()

summarize trainable variables






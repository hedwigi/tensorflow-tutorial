import tensorflow as tf


class Model(object):

    def __init__(self, params):
        """
        构建Graph，不需要sess
        其中train时候需要提供的、读取的变量，都需要通过添加self.变成attributes来获取
        :param params:
        """
        # --------- PLACEHOLDER --------

        # <n hot> because there may be multiple ones
        self.tag_n_hot = tf.placeholder(tf.int32, shape=[None, params["tag_vocab_size"]], name="tag_n_hot")
        self.gender_n_hot = tf.placeholder(tf.int32, shape=[None, params["gender_vocab_size"]], name="gender_n_hot")
        self.city_n_hot = tf.placeholder(tf.int32, shape=[None, params["city_vocab_size"]], name="city_n_hot")
        self.country_n_hot = tf.placeholder(tf.int32, shape=[None, params["country_vocab_size"]], name="country_n_hot")
        self.word_n_hot = tf.placeholder(tf.int32, shape=[None, params["word_vocab_size"]], name="word_n_hot")
        self.friend_n_hot = tf.placeholder(tf.int32, shape=[None, params["friend_vocab_size"]], name="friend_n_hot")
        self.y_true = tf.convert_to_tensor(tf.placeholder(tf.float32, shape=[None, params["num_classes"]], name="y_true"))

        # ------------ GRAPH - embeddings --------

        tag_embedding_table = tf.Variable(
            tf.random_uniform([params["tag_vocab_size"], params["tag_emb_size"]], -1.0, 1.0), name="tag_embedding_table"
        )
        # same size as tag_n_hot, so if input is 0, corresponding embed is [0,0,0...]
        tag_embed = tf.nn.embedding_lookup(tag_embedding_table, self.tag_n_hot)

        gender_embedding_table = tf.Variable(
            tf.random_uniform([params["gender_vocab_size"], params["gender_emb_size"]], -1.0, 1.0), name="gender_embedding_table"
        )
        gender_embed = tf.nn.embedding_lookup(gender_embedding_table, self.gender_n_hot)

        city_embedding_table = tf.Variable(
            tf.random_uniform([params["city_vocab_size"], params["city_emb_size"]], -1.0, 1.0),
            name="city_embedding_table"
        )
        city_embed = tf.nn.embedding_lookup(city_embedding_table, self.city_n_hot)

        country_embedding_table = tf.Variable(
            tf.random_uniform([params["country_vocab_size"], params["country_emb_size"]], -1.0, 1.0),
            name="country_embedding_table"
        )
        country_embed = tf.nn.embedding_lookup(country_embedding_table, self.country_n_hot)

        # init with w2v?
        word_embedding_table = tf.Variable(
            tf.random_uniform([params["word_vocab_size"], params["word_emb_size"]], -1.0, 1.0),
            name="word_embedding_table"
        )
        word_embed = tf.nn.embedding_lookup(word_embedding_table, self.word_n_hot)

        friend_embedding_table = tf.Variable(
            tf.random_uniform([params["friend_vocab_size"], params["friend_emb_size"]], -1.0, 1.0),
            name="friend_embedding_table"
        )
        friend_embed = tf.nn.embedding_lookup(friend_embedding_table, self.friend_n_hot)

        # flatten feature embeddings
        tag_embed_flat = tf.reshape(tag_embed, [-1, params["tag_vocab_size"] * params["tag_emb_size"]])
        gender_embed_flat = tf.reshape(gender_embed, [-1, params["gender_vocab_size"] * params["gender_emb_size"]])
        city_embed_flat = tf.reshape(city_embed, [-1, params["city_vocab_size"] * params["city_emb_size"]])
        country_embed_flat = tf.reshape(country_embed, [-1, params["country_vocab_size"] * params["country_emb_size"]])
        word_embed_flat = tf.reshape(word_embed, [-1, params["word_vocab_size"] * params["word_emb_size"]])
        friend_embed_flat = tf.reshape(friend_embed, [-1, params["friend_vocab_size"] * params["friend_emb_size"]])

        # concat embeddings
        all_feature_embed = tf.concat([tag_embed_flat, gender_embed_flat, city_embed_flat,
                                      country_embed_flat, word_embed_flat, friend_embed_flat], axis=1)
        all_feature_map = tf.reshape(all_feature_embed, [-1, params["img_size"], params["img_size"], params["num_channels"]])
        all_feature_map = tf.nn.dropout(all_feature_map, params["dropout_rate"])

        # ------------ GRAPH - cnn --------
        # [?, 64, 64, 32]
        layer_conv = self.__create_convolutional_layer(input=all_feature_map,
                                                 num_input_channels=params["num_channels"],
                                                 conv_filter_size=params["filter_size_conv"],
                                                 num_filters=params["num_filters_conv"])

        # [?, 16*16*32]
        layer_flat = self.__create_flatten_layer(layer_conv)

        # [?, 128]，使用relu
        layer_fc1 = self.__create_fc_layer(input=layer_flat,
                                    num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                                    num_outputs=params["fc_layer_size"],
                                    use_relu=True)
        # [?, 2] 输出层
        self.layer_fc2 = self.__create_fc_layer(input=layer_fc1,
                                    num_inputs=params["fc_layer_size"],
                                    num_outputs=params["num_classes"],
                                    use_relu=False)
        # [?, 2]
        self.y_pred = tf.nn.sigmoid(self.layer_fc2, name="y_pred")

        # ---------- EVAL ------------
        top_k = tf.nn.top_k(self.y_pred, params["top_k"])
        self.label_prediction = top_k.indices
        # an embedding matrix for later look_up operation
        onehot_embedding = tf.diag([1.] * params["num_classes"])
        # [batch, k, label_size]
        multiple_oneHot = tf.nn.embedding_lookup(
            onehot_embedding,
            self.label_prediction)
        # [batch, label_size]
        multihot = tf.reduce_sum(multiple_oneHot, axis=1)

        correct = tf.reduce_sum(tf.multiply(multihot, self.y_true), 1)
        all_pos = tf.reduce_sum(self.y_true, axis=1)
        self.recall_k = tf.divide(correct, all_pos)
        self.precision_k = tf.divide(correct, params["top_k"])
        self.f1_k = tf.divide(2 * tf.multiply(self.precision_k, self.recall_k), tf.add(self.precision_k, self.recall_k))

        # ---------- OPTIMIZATION ----------
        # forward & softmax + loss
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.layer_fc2, labels=self.y_true)
        self.loss = tf.reduce_mean(cross_entropy)

        # optimize
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        gvs = self.optimizer.compute_gradients(self.loss)
        clipped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        self.train_op = self.optimizer.apply_gradients(clipped_gvs)

    def train(self, sess, trainset, params, num_iterations):
        """

        :param data: Datasets object (train and valid)
        :param params:
        :param num_iterations:
        :return:
        """

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for i in range(num_iterations):
            x_batch, y_true_batch, uid_fid_data = trainset.next_batch(params["batch_size"])

            feed_dict_tr = {self.tag_n_hot: x_batch.tag_n_hot,
                            self.gender_n_hot: x_batch.gender_n_hot,
                            self.city_n_hot: x_batch.city_n_hot,
                            self.country_n_hot: x_batch.country_n_hot,
                            self.word_n_hot:x_batch.word_n_hot,
                            self.friend_n_hot: x_batch.friend_n_hot,
                            self.y_true: y_true_batch}

            sess.run(self.train_op, feed_dict=feed_dict_tr)

            num_batch_per_epoch = int(trainset.num_examples / params["batch_size"])
            if i % num_batch_per_epoch == 0:
                # val_loss = sess.run(self.loss, feed_dict=feed_dict_val)
                epoch = int(i / num_batch_per_epoch)

                # show progress
                precision, recall, f1, y_true, y_pred = sess.run([self.precision_k, self.recall_k, self.f1_k,
                                                              self.y_true, self.label_prediction],
                                                             feed_dict = feed_dict_tr)

                msg = "Training Epoch {0} ---\nk={1},\nTrain precision%k: {2},\n" \
                      "Train recall%k: {3},\nTrain f1%k: {4}\n"

                print("#----- epoch %d:" % epoch)
                print(" y true batch:\n%s" % y_true)
                print(" y pred batch:\n%s" % y_pred)
                print(msg.format(epoch + 1, params["top_k"], precision, recall, f1))
                saver.save(sess, params["model_dir"] + "/" + params["model_base"], global_step=epoch)

    def predict_single(self, uid):
        """

        :param test_image:
        :param test_label:
        :param params:
        :return:
        """


    def __create_weights(self, shape):
        """

        :param shape: list of int
        :return:
        """
        # 截断的正态分布，每个数字都是(-infinity, mean + 2 * stddev]区间的
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def __create_biases(self, size):
        """

        :param size: int
        :return:
        """
        # 写死的
        return tf.Variable(tf.constant(0.05, shape=[size]))

    def __create_convolutional_layer(self, input,
                                   num_input_channels,
                                   conv_filter_size,
                                   num_filters):
        """

        :param input:
        :param num_input_channels:
        :param conv_filter_size:
        :param num_filters:
        :return:
        """
        weights = self.__create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
        biases = self.__create_biases(num_filters)

        # 输出和输入的height, width是同样大小的，[image_size, image_size, num_filters]
        # 每个点的channels求和后，该维度消失
        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1,1,1,1],
                             padding='SAME')

        # broadcast到整个[image_sizem image_size]的矩阵里面去
        layer += biases

        # size = 1/2 original size, [0.5*image_size, 0.5*image_size, num_filters]
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

        layer = tf.nn.relu(layer)
        return layer

    def __create_flatten_layer(self, layer):
        """

        :param layer: [?, size, size, num_filter]
        :return:
        """
        layer_shape = layer.get_shape()

        # 打平所有的feature maps
        num_features = layer_shape[1:4].num_elements()

        layer = tf.reshape(layer, [-1, num_features])
        return layer

    def __create_fc_layer(self, input,
                        num_inputs,
                        num_outputs,
                        use_relu=True):
        """

        :param input:
        :param num_inputs:
        :param num_outputs:
        :param use_relu:
        :return:
        """
        weights = self.__create_weights(shape=[num_inputs, num_outputs])
        biases = self.__create_biases(num_outputs)

        # 右乘weights
        layer = tf.matmul(input, weights) + biases
        if use_relu:
            layer = tf.nn.relu(layer)
        return layer

import tensorflow as tf


class CNNModel(object):

    def __init__(self, params):
        """
        构建Graph，不需要sess
        其中train时候需要提供的、读取的变量，都需要通过添加self.变成attributes来获取
        :param params:
        """
        # --------- PLACEHOLDER --------
        # 因为train时候需要通过feed_dict={ph1:xxx, ph2:xxx}的方式赋值，所以需要self.
        self.x = tf.placeholder(tf.float32,
                           shape=[None, params["img_size"], params["img_size"], params["num_channels"]],
                           name="x")
        self.y_true = tf.placeholder(tf.float32, shape=[None, params["num_classes"]], name="y_true")
        self.y_true_cls = tf.argmax(self.y_true, axis=1)

        # ------------ GRAPH --------
        # [?, 64, 64, 32]
        layer_conv1 = self.__create_convolutional_layer(input=self.x,
                                                 num_input_channels=params["num_channels"],
                                                 conv_filter_size=params["filter_size_conv1"],
                                                 num_filters=params["num_filters_conv1"])
        # [?, 32, 32, 32]
        layer_conv2 = self.__create_convolutional_layer(input=layer_conv1,
                                                 num_input_channels=params["num_filters_conv1"],
                                                 conv_filter_size=params["filter_size_conv2"],
                                                 num_filters=params["num_filters_conv2"])
        # [?, 16, 16, 32]
        layer_conv3 = self.__create_convolutional_layer(input=layer_conv2,
                                                 num_input_channels=params["num_filters_conv2"],
                                                 conv_filter_size=params["filter_size_conv3"],
                                                 num_filters=params["num_filters_conv3"])

        # [?, 16*16*32]
        layer_flat = self.__create_flatten_layer(layer_conv3)

        # [?, 128]，使用relu
        layer_fc1 = self.__create_fc_layer(input=layer_flat,
                                    num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                                    num_outputs=params["fc_layer_size"],
                                    use_relu=True)
        # [?, 2] 输出层，直接使用softmax
        self.layer_fc2 = self.__create_fc_layer(input=layer_fc1,
                                    num_inputs=params["fc_layer_size"],
                                    num_outputs=params["num_classes"],
                                    use_relu=False)
        # [?, 2]
        self.y_pred = tf.nn.softmax(self.layer_fc2, name="y_pred")

        # ---------- EVAL ------------
        # [?,]
        self.y_pred_cls = tf.argmax(self.y_pred, axis=1)

        # current eval
        correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)
        # (pred == true的和) /（总数），等同于计算平均值，平均每个sample得了多少分
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # ---------- OPTIMIZATION ----------
        # forward & loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.layer_fc2, labels=self.y_true)
        self.loss = tf.reduce_mean(cross_entropy)

        # optimize
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)

    def train(self, data, params, num_iterations):
        """

        :param data: Datasets object (train and valid)
        :param params:
        :param num_iterations:
        :return:
        """

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for i in range(num_iterations):
            x_batch, y_true_batch, _, cls_batch = data.train.next_batch(params["batch_size"])
            x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(params["batch_size"])

            feed_dict_tr = {self.x: x_batch,
                            self.y_true: y_true_batch}
            feed_dict_val = {self.x: x_valid_batch,
                             self.y_true: y_valid_batch}

            sess.run(self.optimizer, feed_dict=feed_dict_tr)

            num_batch_per_epoch = int(data.train.num_examples / params["batch_size"])
            if i % num_batch_per_epoch == 0:
                # val_loss = sess.run(self.loss, feed_dict=feed_dict_val)
                epoch = int(i / num_batch_per_epoch)

                # show progress
                acc = sess.run(self.accuracy, feed_dict=feed_dict_tr)
                val_acc, val_loss = sess.run([self.accuracy, self.loss], feed_dict=feed_dict_val)
                msg = "Training Epoch {0} --- Train Accuracy: {1:>6.1%}, " \
                      "Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
                print(msg.format(epoch + 1, acc, val_acc, val_loss))
                saver.save(sess, params["model_dir"] + "/" + params["model_base"], global_step=epoch)

    def predict(self, test_images, test_labels, params):
        """

        :param test_images:
        :param test_labels:
        :param params:
        :return:
        """
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(params["model_dir"]))
        feed_dict_test = {self.x: test_images, self.y_true: test_labels}
        y_pred, acc = sess.run([self.y_pred, self.accuracy], feed_dict=feed_dict_test)
        return y_pred, acc

    def predict_single(self, test_image, test_label, params):
        """

        :param test_image:
        :param test_label:
        :param params:
        :return:
        """
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(params["model_dir"]))
        test_image = test_image.reshape(1, params["img_size"], params["img_size"], params["num_channels"])
        feed_dict_test = {self.x: test_image, self.y_true: test_label}
        y_pred = sess.run(self.y_pred, feed_dict=feed_dict_test)
        return y_pred

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

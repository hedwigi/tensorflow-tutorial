class Dataset(object):

    def __init__(self, images, labels, img_names, cls):
        self.num_examples = images.shape[0]

        self.images = images
        self.labels = labels
        self.img_names = img_names
        self.cls = cls
        self.epochs_done = 0
        self.index_in_epoch = 0

    def next_batch(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        assert batch_size < self.num_examples

        start = self.index_in_epoch
        self.index_in_epoch += batch_size

        # 若最后不够batch_size个，则全部忽略掉，从头开始
        if self.index_in_epoch > self.num_examples:
            self.epochs_done += 1
            start = 0
            self.index_in_epoch = batch_size

            # shuffle
            # self.images, self.labels, self.img_names, self.cls = shuffle(self.images, self.labels, self.img_names, self.cls)

        end = self.index_in_epoch

        return self.images[start: end], self.labels[start: end], \
                self.img_names[start: end], self.cls[start: end]


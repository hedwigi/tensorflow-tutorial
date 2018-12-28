import random
import numpy as np
from l4_input_embd.FeatureVal import FeatureVal


class Dataset(object):

    num_examples = None
    start = None

    def __init__(self, uid_lfid_list, user_features, feature_names, classes):
        """
            (uid, fid) pair is the double_key
            when constructing batch data, will reuse same uid or fid

        :param uid_lfid_list:
        :param user_features:
        :param feature_names:
        :param classes:
        """
        self.uid_lfid_list = uid_lfid_list
        random.shuffle(self.uid_lfid_list)
        self.num_examples = len(uid_lfid_list)

        # {uid: FeatureVal}
        self.user_features = user_features
        self.feature_names = feature_names
        self.start = 0

        # {fid: np.array([0,0,1,0]), ...}
        self.fid2onehot = {}
        num_classes = len(classes)
        fid2idx = {fid: i for i, fid in enumerate(classes)}
        for fid in classes:
            onehot = np.zeros(num_classes)
            onehot[fid2idx[fid]] = 1
            self.fid2onehot[fid] = onehot

        # init counter
        self.counter = 0

    def next_batch(self, batch_size):
        """

        :param batch_size: int
        :return: x_batch: feat_n_hot = [batch_size, feat_vocab_size]
                y_true_batch: [batch_size, num_classes]
                uid_fid_batch
        """
        # reset start if needed
        if self.start + batch_size >= self.num_examples:
            self.start = 0
        batch_data = self.uid_lfid_list[self.start: self.start + batch_size]
        self.start += batch_size

        x_batch = FeatureVal()
        y_true_batch = []
        for uid, lfid in batch_data:
            # add to x_batch
            for feat in self.feature_names:
                feat_name = feat + "_n_hot"
                if x_batch.__getattribute__(feat_name) is None:
                    x_batch.__setattr__(feat_name, [])
                x_batch.__getattribute__(feat_name).append(self.user_features[uid].__getattribute__(feat_name))

            # construct y_true for uid
            y_true = []
            for fid in lfid:
                y_true.append(self.fid2onehot[fid])
            # onehots [1,0,0,0] + [0,0,1,0] -> multihot [1,0,1,0]
            y_true_batch.append(np.sum(y_true, axis=0))

        # stack x
        for feat in self.feature_names:
            feat_name = feat + "_n_hot"
            x_batch.__setattr__(feat_name, np.stack(x_batch.__getattribute__(feat_name)))

        return x_batch, np.array(y_true_batch), batch_data

import numpy as np
from l4_input_embd.FeatureVal import FeatureVal


class FeatureSet(object):

    def __init__(self):
        self.feature_info = {}
        self.feature_vocab_size = {}
        self.feature_names_set = set([])

    def add_feature_info(self, name, vocab_size, feature_dict):
        self.feature_info[name] = feature_dict
        self.feature_vocab_size[name] = vocab_size
        self.feature_names_set.add(name)

    def get_feature_info(self, key):
        return self.feature_info[key]

    def get_feature_vocab_size(self, key):
        return self.feature_vocab_size[key]

    def get_user_features(self, user_vocab):
        """

        :param user_vocab:
        :return: {uid: FeatureVal}
        """
        user_features = {}
        for uid in user_vocab:
            fv = FeatureVal()
            for name in self.feature_names_set:
                # get feature value
                if uid in self.feature_info[name]:
                    val = self.feature_info[name][uid]
                # default value
                else:
                    val = np.zeros(self.feature_vocab_size[name])
                fv.__setattr__(name + "_n_hot", val)

            user_features[uid] = fv
        return user_features

    def get_feature_names_list(self):
        return list(self.feature_names_set)

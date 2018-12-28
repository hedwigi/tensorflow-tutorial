import numpy as np
from l4_input_embd.Dataset import Dataset


class DataHelper(object):

    @staticmethod
    def get_vocab(file_path):
        """
            uid\tval
        :param file_path:
        :return: list of val string
        """
        vocab = set([])
        with open(file_path, "r") as fin:
            line = fin.readline()
            while line:
                _, val = line.strip().split()
                vocab.add(val)
                line = fin.readline()
        return list(vocab)

    @staticmethod
    def get_word_vocab(file_path, sep):
        """
            uid\tw1 w2..
        :param file_path:
        :return:
        """
        vocab = set([])
        with open(file_path, "r") as fin:
            line = fin.readline()
            while line:
                _, tweet = line.strip().split(sep)
                words = tweet.split(" ")
                vocab.update(words)
                line = fin.readline()
        return list(vocab)

    @staticmethod
    def get_uids_fids(filepath, sep):
        """
            [(uid, fid), (uid, fid)...]
            有重复
        :param filepath:
        :return:
        """
        res = []
        with open(filepath, "r") as fin:
            line = fin.readline()
            while line:
                uid, fid = line.strip().split(sep)
                res.append((uid, fid))
                line = fin.readline()
        return res

    @staticmethod
    def construct_feature_dict(path, vocab, sep):
        """

        :param path:
        :param vocab:
        :return: {uid: np.array([1,0,0,1,0,1])}
        """
        val2idx = {val: i for i, val in enumerate(vocab)}
        vocab_size = len(vocab)

        uid2multihot = {}
        with open(path, "r") as fin:
            line = fin.readline()
            while line:
                uid, val = line.strip().split(sep)
                if uid not in uid2multihot:
                    uid2multihot[uid] = np.zeros(vocab_size)

                idx = val2idx[val]
                uid2multihot[uid][idx] = 1

                line = fin.readline()
        return uid2multihot

    @staticmethod
    def construct_uid_lfid_test(uid_fid_test):
        """

        :param uid_fid_test: [(uid, fid), ...]
        :return: [(uid, lfid), ...]
        """
        uid2lfid = {}
        for uid, fid in uid_fid_test:
            if uid not in uid2lfid:
                uid2lfid[uid] = []
            uid2lfid[uid].append(fid)
        return list(uid2lfid.items())

    @staticmethod
    def construct_words_dict(tweet_path, word_vocab, sep):
        """

        :param tweet_path:
        :param word_vocab:
        :param sep:
        :return:
        """
        val2idx = {val: i for i, val in enumerate(word_vocab)}
        vocab_size = len(word_vocab)

        uid2multihot = {}
        with open(tweet_path, "r") as fin:
            line = fin.readline()
            while line:
                uid, tweet = line.strip().split(sep)
                words = tweet.split(" ")
                if uid not in uid2multihot:
                    uid2multihot[uid] = np.zeros(vocab_size)

                for word in words:
                    idx = val2idx[word]
                    uid2multihot[uid][idx] = 1

                line = fin.readline()
        return uid2multihot

import os
import random
import numpy as np
import tensorflow as tf
from l4_input_embd.DataHelper import DataHelper
from l4_input_embd.Dataset import Dataset
from l4_input_embd.FeatureSet import FeatureSet
from l4_input_embd.config import params
from l4_input_embd.Model import Model


# ---- PARAMS ----
dirdata = os.path.join(os.path.dirname(__file__), "data")
validation_size = 0.2

sep = "\t"
train_path = os.path.join(dirdata, "social.train")
uid_fid_train = DataHelper.get_uids_fids(train_path, sep)

# train_y = f_train + f_valid
test_path = os.path.join(dirdata, "social.valid")
uid_fid_test = DataHelper.get_uids_fids(test_path, sep)
uid_fid_test += uid_fid_train

# train should contain all users and all friends
user_vocab = list(set([uid for uid, _ in uid_fid_train]))
friend_vocab = list(set([fid for _, fid in uid_fid_train]))
classes = friend_vocab

params["num_classes"] = len(classes)
params["batch_size"] = 4
params["num_channels"] = 1

# ----- FEATURES ----
# tag
params["tag_vocab_size"] = 13
params["tag_emb_size"] = 3
tag_path = os.path.join(dirdata, "tags.txt")
tag_vocab = DataHelper.get_vocab(tag_path)

# gender
params["gender_vocab_size"] = 2
params["gender_emb_size"] = 2
gender_path = os.path.join(dirdata, "gender.txt")
gender_vocab = DataHelper.get_vocab(gender_path)

# city
params["city_vocab_size"] = 3
params["city_emb_size"] = 8
city_path = os.path.join(dirdata, "city.txt")
city_vocab = DataHelper.get_vocab(city_path)

# country
params["country_vocab_size"] = 5
params["country_emb_size"] = 4
country_path = os.path.join(dirdata, "country.txt")
country_vocab = DataHelper.get_vocab(country_path)

# tweet
params["word_vocab_size"] = 54
params["word_emb_size"] = 32
tweet_path = os.path.join(dirdata, "tweets.txt")
word_vocab = DataHelper.get_word_vocab(tweet_path, sep)

# friend(label embedding)
params["friend_vocab_size"] = 14
params["friend_emb_size"] = 15

feature_set = FeatureSet()
feature_set.add_feature_info("tag", len(tag_vocab), DataHelper.construct_feature_dict(tag_path, tag_vocab, sep))
feature_set.add_feature_info("gender", len(gender_vocab), DataHelper.construct_feature_dict(gender_path, gender_vocab, sep))
feature_set.add_feature_info("city", len(city_vocab), DataHelper.construct_feature_dict(city_path, city_vocab, sep))
feature_set.add_feature_info("country", len(country_vocab), DataHelper.construct_feature_dict(country_path, country_vocab, sep))
feature_set.add_feature_info("word", len(word_vocab), DataHelper.construct_words_dict(tweet_path, word_vocab, sep))
feature_set.add_feature_info("friend", len(friend_vocab), DataHelper.construct_feature_dict(train_path, friend_vocab, sep))


params["img_size"] = 45

# ---- LOAD DATA ----
user_features = feature_set.get_user_features(user_vocab)
feature_names = feature_set.get_feature_names_list()
uid_lfid_test = DataHelper.construct_uid_lfid_test(uid_fid_test)
trainset = Dataset(uid_lfid_test, user_features, feature_names, classes)
x_batch, y_true_batch, batch_data = trainset.next_batch(params["batch_size"])
# print(y_true_batch)

# ---- MAIN -----

mode = "train"
model = Model(params)
sess = tf.Session()

# if mode == "train":
model.train(sess, trainset, params, num_iterations=20)

# else:

    # idx_uid = random.randint(0, len(user_vocab))
    # y_pred = model.predict_single(user_vocab[idx_uid])

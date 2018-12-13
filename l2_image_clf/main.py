import tensorflow as tf
import os
import numpy as np
import random
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from l2_image_clf.DataHelper import DataHelper
from l2_image_clf.entity.Model import Model
from l2_image_clf.config import params


# ---- PARAMS ----
train_path = "./training_data"
classes = os.listdir(train_path)
validation_size = 0.2
test_path = "./test_data"

params["num_classes"] = len(classes)
params["batch_size"] = 32
params["img_size"] = 128
params["num_channels"] = 3

# ---- LOAD DATA ----
train_datasets = DataHelper.read_train_val_sets(train_path, params["img_size"],
                                                classes, validation_size=validation_size)

print("Number of files in Training-set:\t\t{}".format(len(train_datasets.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(train_datasets.valid.labels)))

test_images, test_labels, test_img_names, test_cls = DataHelper.read_test_set(test_path, params["img_size"], classes)
print("Number of files in Test-set:\t{}".format(len(test_images)))

# ---- MAIN -----

mode = "single"
model = Model(params)

if mode == "train":
    model.train(train_datasets, params, num_iterations=3000)

elif mode == "pred_batch":

    acc = model.predict(test_images, test_labels, params)
    print(acc)

else:

    idx_image = random.randint(0, len(test_images))
    y_pred = model.predict_single(np.array([test_images[idx_image]]),
                                  np.array([test_labels[idx_image]]),
                                  params)
    print("Predict image {0} of class {1}".format(test_img_names[idx_image], test_cls[idx_image]))
    print(y_pred)


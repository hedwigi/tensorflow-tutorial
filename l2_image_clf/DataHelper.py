import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from l2_image_clf.entity.Datasets import Datasets
from l2_image_clf.entity.Dataset import Dataset


class DataHelper(object):

    @staticmethod
    def read_train_val_sets(train_path, image_size, classes, validation_size):
        """

        :param train_path:
        :param image_size:
        :param classes:
        :param validation_size:
        :return:
        """
        images, labels, img_names, cls = DataHelper.__load_data(train_path, image_size, classes)
        images, labels, img_names, cls = shuffle(images, labels, img_names, cls)

        if isinstance(validation_size, float):
            validation_size = int(validation_size * images.shape[0])

        val_images = images[:validation_size]
        val_labels = labels[:validation_size]
        val_img_names = img_names[:validation_size]
        val_cls = cls[:validation_size]

        train_images = images[validation_size:]
        train_labels = labels[validation_size:]
        train_img_names = img_names[validation_size:]
        train_cls = cls[validation_size:]

        datasets = Datasets()
        datasets.train = Dataset(train_images, train_labels, train_img_names, train_cls)
        datasets.valid = Dataset(val_images, val_labels, val_img_names, val_cls)
        return datasets

    @staticmethod
    def read_test_set(test_path, image_size, classes):
        """

        :param test_path:
        :param image_size:
        :param classes:
        :return:
        """
        images, labels, img_names, cls = DataHelper.__load_data(test_path, image_size, classes)
        return images, labels, img_names, cls

    @staticmethod
    def __load_data(data_path, image_size, classes):
        """

        :param data_path:
        :param image_size:
        :param classes:
        :return:
        """
        images = []
        labels = []
        img_names = []
        cls = []

        for class_ in classes:
            index = classes.index(class_)
            print("reading {} files (Index: {})".format(class_, index))
            dir = os.path.join(data_path, class_)
            files = os.listdir(dir)
            for file in files:
                # image
                image = cv2.imread(os.path.join(dir, file))
                image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
                image = image.astype(np.float32)
                image = np.multiply(image, 1.0/255.0)
                images.append(image)

                # label
                label = np.zeros(len(classes))
                label[index] = 1.0
                labels.append(label)

                img_names.append(os.path.basename(file))
                cls.append(class_)
        return np.array(images), np.array(labels), np.array(img_names), np.array(cls)

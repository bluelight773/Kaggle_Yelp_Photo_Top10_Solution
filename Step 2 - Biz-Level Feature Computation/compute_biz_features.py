"""Compute business-level features by averaging the features computed for each business' images in the previous step.

Computed features are based on arithmetic mean of the features extracted in the previous step and are saved in .csv
files, so they can be used by step 3 predictor script.

Ensure data_root, do_train, do_test are all set as desired before running.

data_root is the full path containing (amongst other things) the .h5 files in which the image-level features are stored.

This file is largely based on:
https://github.com/ncchen55414/Kaggle-Yelp/tree/master/CNN_Submission1
"""
import time

import h5py
import numpy as np
import pandas as pd

data_root = '/path/to/data_root/'

do_train = True
do_test = True


class FeatureSet(object):
    """Abstraction for a set of features of a particular type (eg inception-based vs vggplaces-based)"""

    def __init__(self, train_img_features_h5, test_img_features_h5, layers,
                 train_biz_csv_fmt_str, test_biz_csv_fmt_str):
        self.train_img_features_h5 = train_img_features_h5
        self.test_img_features_h5 = test_img_features_h5
        self.layers = layers
        self.train_biz_csv_fmt_str = train_biz_csv_fmt_str
        self.test_biz_csv_fmt_str = test_biz_csv_fmt_str


inception_feature_set = FeatureSet(data_root+'train_image_inc_features.h5', data_root+'test_image_inc_features.h5',
                                   ("p3", "sm"),
                                   data_root + "train_biz_inc_{}features.csv",
                                   data_root + "test_biz_inc_{}features.csv")

vggplaces_feature_set = FeatureSet(data_root + 'train_image_vggplaces_features.h5',
                                   data_root + 'test_image_vggplaces_features.h5',
                                   ("fc6", "fc7"),
                                   data_root + "train_biz_vggplaces_{}features.csv",
                                   data_root + "test_biz_vggplaces_{}features.csv")


for feature_set in (inception_feature_set, vggplaces_feature_set):
    for layer in feature_set.layers:
        if do_train:
            train_photo_to_biz = pd.read_csv(data_root+'train_photo_to_biz_ids.csv')
            train_labels = pd.read_csv(data_root+'train.csv').dropna()
            train_labels['labels'] = train_labels['labels'].apply(lambda x: tuple(sorted(int(t) for t in x.split())))
            train_labels.set_index('business_id', inplace=True)
            biz_ids = train_labels.index.unique()
            print("Number of business: {} (4 businesses with missing labels are dropped)".format(len(biz_ids)))

            # Load image features
            f = h5py.File(feature_set.train_img_features_h5, 'r')
            train_image_features = f['{} feature'.format(layer)]

            t = time.time()

            # For each business, compute a feature vector
            df = pd.DataFrame(columns=['business', 'label', '{} feature vector'.format(layer)])

            index = 0
            for biz in biz_ids:

                label = train_labels.loc[biz]['labels']
                image_index = train_photo_to_biz[train_photo_to_biz['business_id'] == biz].index.tolist()

                features = train_image_features[image_index]
                stat_feature = list(np.mean(features, axis=0))

                df.loc[index] = [biz, label, stat_feature]
                index += 1
                if index % 100 == 0:
                    print("Business processed: ", index, "Time passed: ", "{0:.1f}".format(time.time()-t), "sec")

            out_train_biz_features_csv = feature_set.train_biz_csv_fmt_str.format(layer)
            with open(out_train_biz_features_csv, 'w') as f:
                df.to_csv(f, index=False)

            # Check file content
            train_business = pd.read_csv(out_train_biz_features_csv)
            print(train_business.shape)
            print(train_business[0:5])
            f.close()

        if do_test:
            test_photo_to_biz = pd.read_csv(data_root+'test_photo_to_biz.csv')
            biz_ids = test_photo_to_biz['business_id'].unique()

            # Load image features
            f = h5py.File(feature_set.test_img_features_h5, 'r')
            image_filenames = list(np.copy(f['photo_id']))
            # remove the full path and the str ".jpg"
            image_filenames = [name.split('/')[-1][:-4] for name in image_filenames]
            image_features = f['{} feature'.format(layer)]

            print("Number of business: {}".format(len(biz_ids)))

            df = pd.DataFrame(columns=['business', 'feature vector'])
            index = 0
            t = time.time()

            for biz in biz_ids:

                image_ids = test_photo_to_biz[test_photo_to_biz['business_id'] == biz]['photo_id'].tolist()
                image_index = sorted([image_filenames.index(str(x)) for x in image_ids])

                features = image_features[image_index]
                stat_feature = list(np.mean(features, axis=0))
                df.loc[index] = [biz, stat_feature]
                index += 1
                if index % 100 == 0:
                    print("Business processed: {}. Time passed: {0:.1f} sec".format(index, time.time()-t))

            out_test_biz_features_csv = feature_set.test_biz_csv_fmt_str.format(layer)
            with open(out_test_biz_features_csv, 'w') as f:
                df.to_csv(f, index=False)

            # Check file content
            test_business = pd.read_csv(out_test_biz_features_csv)
            print(test_business.shape)
            print(test_business[0:5])
            f.close()

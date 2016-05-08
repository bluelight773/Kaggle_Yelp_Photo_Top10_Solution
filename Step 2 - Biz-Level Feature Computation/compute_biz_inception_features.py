"""Compute business-level features by averaging the inception features computed for its images in the previous step.

Computed features are based on arithmetic mean of the features extracted in the previous step and are saved in .csv
files, so they can be used by step 3 predictor script.

Ensure data_root, do_train, do_test are all set as desired before running.

data_root is the full path containing:
1) all the CSVs provided as the competition data
2) train_photos folder with all the training photos inside
3) test_photos folder with all the test photos inside

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

LAYERS = ['p3', 'sm']

TRAINED_H5 = data_root+'train_image_inc_features.h5'
TESTED_H5 = data_root+'test_image_inc_features.h5'


for LAYER in LAYERS:

    OUT_TRAIN_BIZ_FEATURES_CSV = data_root+"train_biz_inc_{}features.csv".format(LAYER)
    OUT_TEST_BIZ_FEATURES_CSV = data_root+"test_biz_inc_{}features.csv".format(LAYER)

    if do_train:
        train_photo_to_biz = pd.read_csv(data_root+'train_photo_to_biz_ids.csv')
        train_labels = pd.read_csv(data_root+'train.csv').dropna()
        train_labels['labels'] = train_labels['labels'].apply(lambda x: tuple(sorted(int(t) for t in x.split())))
        train_labels.set_index('business_id', inplace=True)
        biz_ids = train_labels.index.unique()
        print("Number of business: ", len(biz_ids), "(4 businesses with missing labels are dropped)")

        # Load image features
        f = h5py.File(TRAINED_H5, 'r')
        train_image_features = f['{} feature'.format(LAYER)]

        t = time.time()

        # For each business, compute a feature vector
        df = pd.DataFrame(columns=['business', 'label', 'inc{} feature vector'.format(LAYER)])

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

        with open(OUT_TRAIN_BIZ_FEATURES_CSV, 'w') as f:
            df.to_csv(f, index=False)

        # Check file content
        train_business = pd.read_csv(OUT_TRAIN_BIZ_FEATURES_CSV)
        print(train_business.shape)
        print(train_business[0:5])
        f.close()

    if do_test:
        test_photo_to_biz = pd.read_csv(data_root+'test_photo_to_biz.csv')
        biz_ids = test_photo_to_biz['business_id'].unique()

        # Load image features
        f = h5py.File(TESTED_H5, 'r')
        image_filenames = list(np.copy(f['photo_id']))
        # remove the full path and the str ".jpg"
        image_filenames = [name.split('/')[-1][:-4] for name in image_filenames]
        image_features = f['{} feature'.format(LAYER)]

        print("Number of business: ", len(biz_ids))

        df = pd.DataFrame(columns=['business', 'inc{} feature vector'.format(LAYER)])
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
                print("Business processed: ", index, "Time passed: ", "{0:.1f}".format(time.time()-t), "sec")

        with open(OUT_TEST_BIZ_FEATURES_CSV, 'w') as f:
            df.to_csv(f, index=False)

        # Check file content
        test_business = pd.read_csv(OUT_TEST_BIZ_FEATURES_CSV)
        print(test_business.shape)
        print(test_business[0:5])
        f.close()

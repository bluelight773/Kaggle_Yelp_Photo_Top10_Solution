"""Extract image features from the last 2 layers of the Inception model, trained on ImageNet 2012 Challenge data set.

Extracted features are saved in .h5 files, so that business-level features can be computed from that data in the next
step.

Ensure model_dir, data_root, do_train, do_test are all set as desired before running.

model_dir needs to include files pertaining to the Inception model.  These files can be obtained by downloading and
extracting:  http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz

data_root is the full path containing:
1) all the CSVs provided as the competition data
2) train_photos folder with all the training photos inside
3) test_photos folder with all the test photos inside

This file is largely based on:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/imagenet/classify_image.py
https://github.com/ncchen55414/Kaggle-Yelp/tree/master/CNN_Submission1

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import os.path

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', '/path/to/model_dir',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")

data_root = '/path/to/data_root/'

# Specify whether to extract features for training images, test images, or both.
do_train = True
do_test = True


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            tf.app.flags.FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def train():
    """Extract training image features, based on the last 2 layers (pool3:0 and softmax:0) and save to .h5."""

    # Set up the initial .h5 file if it doesn't already exist.
    if not os.path.isfile(data_root + 'train_image_inc_features.h5'):
        f = h5py.File(data_root + 'train_image_inc_features.h5', 'w')
        filenames = f.create_dataset('photo_id', (0,), maxshape=(None,), dtype='|S54')
        sm_feature = f.create_dataset('sm feature', (0, 1008), maxshape=(None, 1008))
        feature_p3 = f.create_dataset('p3 feature', (0, 2048), maxshape=(None, 2048))
        f.close()

    f = h5py.File(data_root + 'train_image_inc_features.h5', 'r+')
    images_already_done = np.copy(f['photo_id'])
    f.close()

    train_photos = pd.read_csv(data_root + 'train_photo_to_biz_ids.csv')
    train_folder = data_root + 'train_photos/'
    train_images = [os.path.join(train_folder, str(x) + '.jpg') for x in train_photos['photo_id']]  # get full filename

    num_train = len(train_images)
    num_done = 0

    # Extract features per image while attempting to not repeat extraction of features for images covered before
    # ie: images that are already inside the .h
    with tf.Session() as sess:
        for i in range(num_train):
            image = train_images[i]

            if image in images_already_done:
                num_done += 1
                if num_done % 1000 == 0 or num_done == num_train:
                    print("Train images processed: {}".format(num_done))
                continue

            if not tf.gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)
            image_data = tf.gfile.FastGFile(image, 'rb').read()

            # Extract features from the last layer
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
            features_sm = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            features_sm = np.squeeze(features_sm)

            # Extract features from the second-last layer
            p3_tensor = sess.graph.get_tensor_by_name('pool_3:0')
            features_p3 = sess.run(p3_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            features_p3 = np.squeeze(features_p3)

            num_done += 1
            f = h5py.File(data_root + 'train_image_inc_features.h5', 'r+')
            f['photo_id'].resize((num_done,))
            f['photo_id'][i: num_done] = np.array(train_images[i:i + 1])
            f['sm feature'].resize((num_done, len(features_sm)))
            f['sm feature'][i: num_done, :] = features_sm
            f['p3 feature'].resize((num_done, len(features_p3)))
            f['p3 feature'][i: num_done, :] = features_p3
            f.close()
            if num_done % 100 == 0 or num_done == num_train:
                print("Train images processed: {}".format(num_done))

    # Check the file content
    f = h5py.File(data_root + 'train_image_inc_features.h5', 'r')
    print('train_image_inc_features.h5:')
    for key in f.keys():
        print("{}, {}".format(key, f[key].shape))

    print("\nA photo: {}".format(f['photo_id'][0]))
    f.close()


def test():
    """Extract test image features, based on the last 2 layers (pool3:0 and softmax:0) and save to .h5"""

    # Set up the initial .h5 file if it doesn't already exist.
    if not os.path.isfile(data_root + 'test_image_inc_features.h5'):
        f = h5py.File(data_root + 'test_image_inc_features.h5', 'w')
        filenames = f.create_dataset('photo_id', (0,), maxshape=(None,), dtype='|S54')
        sm_feature = f.create_dataset('sm feature', (0, 1008), maxshape=(None, 1008))
        feature_p3 = f.create_dataset('p3 feature', (0, 2048), maxshape=(None, 2048))
        f.close()

    f = h5py.File(data_root + 'test_image_inc_features.h5', 'r+')
    images_already_done = np.copy(f['photo_id'])
    f.close()

    test_photos = pd.read_csv(data_root + 'test_photo_to_biz.csv')
    test_folder = data_root + 'test_photos/'
    test_images = [os.path.join(test_folder, str(x) + '.jpg') for x in test_photos['photo_id'].unique()]
    num_test = len(test_images)
    print("Number of test images: {}".format(num_test))

    # Extract features per image while attempting to not repeat extraction of features for images covered before
    # ie: images that are already inside the .h5.
    num_done = 0
    with tf.Session() as sess:
        for i in range(num_test):
            image = test_images[i]

            if image in images_already_done:
                num_done += 1
                if num_done % 1000 == 0 or num_done == num_test:
                    print("Test images processed: {}".format(num_done))
                continue

            if not tf.gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)
            image_data = tf.gfile.FastGFile(image, 'rb').read()

            # Extract features from the last layer
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
            features_sm = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            features_sm = np.squeeze(features_sm)

            # Extract features from the second-last layer
            p3_tensor = sess.graph.get_tensor_by_name('pool_3:0')
            features_p3 = sess.run(p3_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            features_p3 = np.squeeze(features_p3)

            num_done += 1

            f = h5py.File(data_root + 'test_image_inc_features.h5', 'r+')
            f['photo_id'].resize((num_done,))
            f['photo_id'][i: num_done] = np.array(test_images[i:i + 1])
            f['sm feature'].resize((num_done, len(features_sm)))
            f['sm feature'][i: num_done, :] = features_sm
            f['p3 feature'].resize((num_done, len(features_p3)))
            f['p3 feature'][i: num_done, :] = features_p3
            f.close()
            if num_done % 100 == 0 or num_done == num_test:
                print("Test images processed: {}".format(num_done))

    # Check the file content
    f = h5py.File(data_root + 'test_image_inc_features.h5', 'r')
    for key in f.keys():
        print("{} {}".format(key, f[key].shape))
    print("\nA photo: {}".format(f['photo_id'][0]))
    f.close()


create_graph()
if do_train:
    train()

if do_test:
    test()

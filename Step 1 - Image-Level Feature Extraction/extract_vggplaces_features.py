"""Extract image features from the last two fully-connected layers of FC6 and FC7 in MIT's Places205-VGG Caffe model.

Extracted features are saved in .h5 files, so that business-level features can be computed from that data in the next
step.

Ensure caffe_root, data_root, do_train, do_test are all set as desired before running.  Additionally, depending on
your hardware and load constraints, you may need to experiment with different values of batch_size.

caffe_root is the folder in which caffe is installed, and should include the following:
1) python folder which has Caffe's python-related code
2) models/places205VGG16 folder that contains .caffemodel and .prototxt files pertaining to MIT's Places205-VGG model,
which can be obtained from: http://places.csail.mit.edu/downloadCNN.html

data_root is the full path containing:
1) all the CSVs provided as the competition data
2) train_photos folder with all the training photos inside
3) test_photos folder with all the test photos inside

This file is largely based on:
https://github.com/ncchen55414/Kaggle-Yelp/tree/master/CNN_Submission1

MIT Places CNN Citation:
B. Zhou, A. Lapedriza, J. Xiao, A. Torralba, and A. Oliva
Learning Deep Features for Scene Recognition using Places Database.
Advances in Neural Information Processing Systems 27 (NIPS) spotlight, 2014.
"""

import sys
import os

import caffe
import h5py
import numpy as np
import pandas as pd


caffe_root = '/path/to/caffe/'
data_root = '/path/to/data_root/'

do_train = True
do_test = True

batch_size = 300

LAYERS = ['fc6', 'fc7']
LAYER_SIZES = [4096, 4096]

CAFFE_MODEL = caffe_root + 'models/places205VGG16/snapshot_iter_765280.caffemodel'
CAFFE_PROTOTXT = caffe_root + 'models/places205VGG16/deploy_10.prototxt'

TRAIN_IMAGE_MEAN = np.array((105.487823486, 113.741088867, 116.060394287))
CROP_SIZE = 224

TRAIN_H5 = data_root + 'train_image_vggplaces_features.h5'
TEST_H5 = data_root + 'test_image_vggplaces_features.h5'


sys.path.insert(0, caffe_root + 'python')

# Uncomment to use GPU
# caffe.set_device(0)
# caffe.set_mode_gpu()


def extract_features(images, layers):
    """Extract deep-learning features of provided images based on provided layers"""
    net = caffe.Net(CAFFE_PROTOTXT, CAFFE_MODEL, caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', TRAIN_IMAGE_MEAN)  # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB]]

    num_images = len(images)
    net.blobs['data'].reshape(num_images, 3, CROP_SIZE, CROP_SIZE)
    net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data', caffe.io.load_image(x)), images)
    out = net.forward()
    if len(layers) == 1:
        return [net.blobs[layers[0]].data]
    features_list = []
    for layer in layers:
        features_list.append(net.blobs[layer].data)
    return features_list


def train():
    """Extract training image features, based on layers FC6 and FC7 and save to .h5."""

    # Initialize files

    # f.close()
    f = h5py.File(TRAIN_H5, 'w')
    filenames = f.create_dataset('photo_id', (0,), maxshape=(None,), dtype='|S54')
    for i in range(len(LAYERS)):
        layer_name = LAYERS[i]
        layer_size = LAYER_SIZES[i]
        feature = f.create_dataset('feature_{}'.format(layer_name), (0, layer_size), maxshape=(None, layer_size))
    f.close()

    train_photos = pd.read_csv(data_root + 'train_photo_to_biz_ids.csv')
    train_folder = data_root + 'train_photos/'
    train_images = [os.path.join(train_folder, str(x) + '.jpg') for x in train_photos['photo_id']]  # get full filename

    num_train = len(train_images)
    print "Number of training images: ", num_train

    # Training Images
    for i in range(0, num_train, batch_size):
        images = train_images[i: min(i + batch_size, num_train)]

        features_list = extract_features(images, LAYERS)

        num_done = i + features_list[0].shape[0]
        f = h5py.File(TRAIN_H5, 'r+')
        f['photo_id'].resize((num_done,))
        f['photo_id'][i: num_done] = np.array(images)
        for j in range(len(features_list)):
            features = features_list[j]
            f['feature_{}'.format(LAYERS[j])].resize((num_done, features.shape[1]))
            f['feature_{}'.format(LAYERS[j])][i: num_done, :] = features
        f.close()
        if num_done % batch_size == 0 or num_done == num_train:
            print "Train images processed: ", num_done

    # Check the file content
    f = h5py.File(TRAIN_H5, 'r')
    print TRAIN_H5
    for key in f.keys():
        print key, f[key].shape

    print "\nA photo:", f['photo_id'][0]
    for i in range(len(LAYERS)):
        print "A feature vector (first 10-dim): ", f['feature_{}'.format(LAYERS[i])][0][0:10], " ..."
    f.close()


def test():
    """Extract test image features, based on layers FC6 and FC7 and save to .h5."""

    f = h5py.File(TEST_H5, 'w')
    filenames = f.create_dataset('photo_id', (0,), maxshape=(None,), dtype='|S54')
    for i in range(len(LAYERS)):
        layer_name = LAYERS[i]
        layer_size = LAYER_SIZES[i]
        feature = f.create_dataset('feature_{}'.format(layer_name), (0, layer_size), maxshape=(None, layer_size))
    f.close()

    test_photos = pd.read_csv(data_root + 'test_photo_to_biz.csv')
    test_folder = data_root + 'test_photos/'
    test_images = [os.path.join(test_folder, str(x) + '.jpg') for x in test_photos['photo_id'].unique()]
    num_test = len(test_images)
    print "Number of test images: ", num_test

    # Test Images
    for i in range(0, num_test, batch_size):
        images = test_images[i: min(i + batch_size, num_test)]
        features_list = extract_features(images, LAYERS)
        num_done = i + features_list[0].shape[0]

        f = h5py.File(TEST_H5, 'r+')
        f['photo_id'].resize((num_done,))
        f['photo_id'][i: num_done] = np.array(images)
        for j in range(len(features_list)):
            features = features_list[j]
            f['feature_{}'.format(LAYERS[j])].resize((num_done, features.shape[1]))
            f['feature_{}'.format(LAYERS[j])][i: num_done, :] = features
        f.close()
        if num_done % batch_size == 0 or num_done == num_test:
            print "Test images processed: ", num_done

    # Check the file content
    f = h5py.File(TEST_H5, 'r')
    print TEST_H5
    for key in f.keys():
        print key, f[key].shape
    print "\nA photo:", f['photo_id'][0]
    for i in range(len(LAYERS)):
        print "A feature vector (first 10-dim): ", f['feature_{}'.format(LAYERS[i])][0][0:10], " ..."
    f.close()

if do_train:
    train()

if do_test:
    test()

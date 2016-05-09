"""Build a classifier and make predictions based on features in CSVs generated in the previous step.

Prediction results are outputted in a submission.csv that could be used for Kaggle competition submissions.

Ensure data_root is set correctly.

data_root is the full path containing (amongst other things) the CSVs with the business-level features, generated in
the previous step.

This file is largely based on:
https://github.com/ncchen55414/Kaggle-Yelp/tree/master/CNN_Submission1
"""
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

import numpy as np
import pandas as pd

data_root = '/path/to/data_root/'

SEED = 100

train_biz_features_files = (data_root + "train_biz_inc_p3features.csv",
                            data_root + "train_biz_inc_smfeatures.csv",
                            data_root + "train_biz_vggplaces_fc6features.csv",
                            data_root + "train_biz_vggplaces_fc7features.csv"
                            )

test_biz_features_files = (data_root + "test_biz_inc_p3features.csv",
                           data_root + "test_biz_inc_smfeatures.csv",
                           data_root + "test_biz_vggplaces_fc6features.csv",
                           data_root + "test_biz_vggplaces_fc7features.csv"
                           )

feature_vector_names = ("p3 feature vector", "sm feature vector",
                        "fc6 features vector", "fc7 features vector")

out_file = data_root + "submission.csv"


# Build a training data frame that contains all the features computed in the previous step per business
for i in range(len(train_biz_features_files)):
    new_df = pd.read_csv(train_biz_features_files[i])
    if i == 0:
        train_df = new_df
    else:
        new_df = new_df.drop('label', 1)
        train_df = pd.merge(train_df, new_df)


# Build a test data frame that contains all the features computed in the previous step per business
for i in range(len(test_biz_features_files)):
    new_df = pd.read_csv(test_biz_features_files[i])
    if i == 0:
        test_df = new_df
    else:
        test_df = pd.merge(test_df, new_df)


def convert_label_to_array(str_label):
    str_label = str_label[1:-1]
    str_label = str_label.split(',')
    return [int(x) for x in str_label if len(x) > 0]


def convert_feature_to_vector(str_feature):
    str_feature = str_feature[1:-1]
    str_feature = str_feature.split(',')
    return [float(x) for x in str_feature]

y_train = np.array([convert_label_to_array(y) for y in train_df['label']])

# Ensure features are stored as numpy arrays that could easily be transformed and used for classification purposes
for i in range(len(feature_vector_names)):
    fvn = feature_vector_names[i]
    print(fvn)
    if i == 0:
        X_train = np.array([convert_feature_to_vector(x) for x in train_df[fvn]])
        X_test = np.array([convert_feature_to_vector(x) for x in test_df[fvn]])
    else:
        X_train = np.concatenate((X_train, np.array([convert_feature_to_vector(x) for x in train_df[fvn]])), axis=1)
        X_test = np.concatenate((X_test, np.array([convert_feature_to_vector(x) for x in test_df[fvn]])), axis=1)


# Normalize to [0, 1] range based on training data
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

print("X_train: {}".format(X_train.shape))
print("y_train: {}".format(y_train.shape))
print("X_test: {}".format(X_test.shape))
print("train_df:")
print(train_df[0:5])

print("Normed X_train")
print(X_train[0:5, 0:5])

t = time.time()

mlb = MultiLabelBinarizer()
y_ptrain = mlb.fit_transform(y_train)  # Convert list of labels to binary matrix

random_state = np.random.RandomState(SEED)
X_ptrain, X_ptest, y_ptrain, y_ptest = train_test_split(X_train, y_ptrain, test_size=.2, random_state=random_state)

print("About to start training classifier with set parameters on subset of train data")
classifier = OneVsRestClassifier(GradientBoostingClassifier(learning_rate=0.01, n_estimators=5000, subsample=0.5,
                                                            min_samples_split=175, min_samples_leaf=10, max_depth=5,
                                                            max_features='sqrt',
                                                            verbose=1,
                                                            random_state=SEED))
classifier.fit(X_ptrain, y_ptrain)

print("About to make predictions on sample of training data")
y_ppredict = classifier.predict(X_ptest)

print("Time passed: {0:.1f} sec".format(time.time() - t))
print("Samples of predicted labels (in binary matrix):\n{}".format(y_ppredict[0:3]))
print("\nSamples of predicted labels:\n", mlb.inverse_transform(y_ppredict[0:3]))
statistics = pd.DataFrame(columns=["attribute " + str(i) for i in range(9)] + ['num_biz'],
                          index=["biz count", "biz ratio"])
pd.options.display.float_format = '{:.0f}%'.format
print(statistics)
print("F1 score: {}".format(f1_score(y_ptest, y_ppredict, average='micro')))
print("Individual Class F1 score: {}".format(f1_score(y_ptest, y_ppredict, average=None)))


# Re-Train classifier using all training data, and make predictions on test set
t = time.time()

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train)  # Convert list of labels to binary matrix

print("About to train classifier on all training data (to have it ready to predict on submission test data)")
classifier.fit(X_train, y_train)

print("About to make prediction for submission test data")
y_predict = classifier.predict(X_test)

y_predict_label = mlb.inverse_transform(y_predict)  # Convert binary matrix back to labels

print("Time passed: {0:.1f} sec".format(time.time() - t))
print(X_test.shape)

test_data_frame = test_df
df = pd.DataFrame(columns=['business_id', 'labels'])

for i in range(len(test_data_frame)):
    biz = test_data_frame.loc[i]['business']
    label = y_predict_label[i]
    label = str(label)[1:-1].replace(",", " ")
    df.loc[i] = [str(biz), label]

with open(out_file, 'w') as f:
    df.to_csv(f, index=False)

statistics = pd.DataFrame(columns=["attribute " + str(i) for i in range(9)] + ['num_biz'],
                          index=["biz count", "biz ratio"])

pd.options.display.float_format = '{:.0f}%'.format
print(statistics)

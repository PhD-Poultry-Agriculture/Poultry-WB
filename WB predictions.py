# predicts the correct score of WB
# 8 control animals(per time point) 8 WB: 16 (8, 8) for tp1, 16 for tp2,..16 for tp4
#%%
def getline(animalnum, arr): # for a give animal gets all the assosiated markers values, t1-t4
    lst0 = arr.tolist()
    animaldata = []
    for ii in range(animalnum, 64, 16):
        animaldata = animaldata + lst0[ii]
    return animaldata


from scipy.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn import metrics
import statistics
import pandas as pd
import numpy as np
import numpy.random
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn import datasets
from sklearn.model_selection import KFold
import math
from sklearn.model_selection import LeaveOneOut

# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

datasetF1 = pd.read_csv('WB markers polar', header = None) # every row is a marker, every column animal, 4 time points, with periodicity of 16 ( 8 cont, 8 WB)
print(datasetF1.head()) # [5 rows x 64 columns]
#             0            1   ...            62            63
# 0  8.183752e+06  12341236.63  ...  7.070200e+06  7.101349e+06
# 1  1.365956e+07  20548241.03  ...  1.631861e+07  1.633326e+07
# 2  4.903491e+07  88683021.51  ...  1.877162e+08  3.053277e+08
# 3  1.561539e+07  27098603.99  ...  5.057461e+06  8.085194e+06
# 4  1.937965e+07  17341629.74  ...  2.001933e+07  2.459709e+07

# extract data for animals: each row will be an animal, then values according to the 27 markers for a given timepoint, then 27 markers and next timepoint
# transpose
arr0 = datasetF1.iloc[:, :].values
arr0t = arr0.transpose()
print(arr0t[0,0], arr0t[0,1], len(arr0t), len(arr0t[0])) # 8183752.319, 13659557.66, 64 (lines), 27 (markers) - yey
# for first animal get lines: 0, 16, 32, 48
animal1 = getline(0, arr0t) # extract data for first animal
print(animal1, len(animal1)) # 108 (27*4) 27 markers, 4 timepoints - yey!
# first value of aimal1 at timepoint2 (# 27, starts from 0), (of the 4) should be first value of line 16 in arr0t (first line is line0)
print(animal1[27], arr0t[16,0]) # 11298827.81 11298827.81 - yey!
# now for all animals
animals0 = []
for ii in range(0, 16):
    tmp = getline(ii, arr0t)
    animals0 = animals0 + [tmp]
print(len(animals0), len(animals0[0]), len(animals0[1])) # 16 108 108
# animal2, first value, or firat value at timepoint 2
print(animals0[1][0], arr0t[1,0]) # 12341236.63 12341236.63
print(animals0[1][27], arr0t[17,0]) # 4200985.092 4200985.092 - yey!

#%% add 0 as control group and 1 as WB
gpcont = [0] * 8
gpwb = [1] * 8
gptot = gpcont + gpwb
animals0t = np.array(animals0).transpose()
print(len(animals0t), len(animals0t[0])) # 108 16 - yey
animalslst = animals0t.tolist()
animalslstgp = animalslst + [gptot]
print(len(animalslstgp), len(animalslstgp[0]), len(animalslstgp[-1])) # 109 16 16
animalsgparr = np.array(animalslstgp).transpose()
print(len(animalsgparr), len(animalsgparr[0]), len(animalsgparr[-1])) # 16 109 109 - yey

# also permute the labels to control for the 100% accuracy in the classifier
permlabels = np.random.permutation(gptot)
print('permlabels', permlabels)
# permlabels [0 0 0 1 1 1 0 0 0 1 1 1 0 0 1 1]
# now use permlabels and test the classifier
animalslstperm = animalslst + [permlabels]
print(len(animalslstperm), len(animalslstperm[0]), len(animalslstgp[-1])) # 109 16 16
animalspermarr = np.array(animalslstperm).transpose()
print(len(animalspermarr), len(animalspermarr[0]), len(animalspermarr[-1])) # 16 109 109


# applying random forest classifier
X = animalsgparr[:, 0:-1]
Y = animalsgparr[:, -1]
classifier = RandomForestClassifier(random_state=0)

cvout = LeaveOneOut()
# # enumerate splits
y_true, y_pred = list(), list()
for train_ix, test_ix in cvout.split(X):
    # print(test_ix)
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = Y[train_ix], Y[test_ix]
    # fit model
    model = RandomForestClassifier(random_state=1)
    model.fit(X_train, y_train)
    # evaluate model
    yhat = model.predict(X_test)
    # store
    y_true.append(y_test[0])
    y_pred.append(yhat[0])

# calculate accuracy
acc = accuracy_score(y_true, y_pred)
print('Accuracy: %.3f' % acc)
# Accuracy: 1.000
print('true, predictions', y_true, y_pred)

# true [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# predictions [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# confusion matrix
cnfmtrx = confusion_matrix(y_true,y_pred)
print(confusion_matrix(y_true,y_pred)) # rows - true, col - prediction
# [[8 0]
#  [0 8]]


cm = confusion_matrix(y_true, y_pred, labels=model.classes_) # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
disp0 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp0.plot()
# plt.savefig('confusion matrix control WB.pdf')
plt.show() #

# with the permuted form
# applying random forest classifier
X1 = animalspermarr[:, 0:-1]
Y1 = animalspermarr[:, -1]
classifier = RandomForestClassifier(random_state=0)

cvout = LeaveOneOut()
# # enumerate splits
y1_true, y1_pred = list(), list()
for train_ix, test_ix in cvout.split(X1):
    # print(test_ix)
    # split data
    X1_train, X1_test = X1[train_ix, :], X1[test_ix, :]
    y1_train, y1_test = Y1[train_ix], Y1[test_ix]
    # fit model
    model = RandomForestClassifier(random_state=1)
    model.fit(X1_train, y1_train)
    # evaluate model
    y1hat = model.predict(X1_test)
    # store
    y1_true.append(y1_test[0])
    y1_pred.append(y1hat[0])

# calculate accuracy
acc1 = accuracy_score(y1_true, y1_pred)
print('Accuracy: %.3f' % acc1)
# Accuracy: 0.125
print('true, predictions', y1_true, y1_pred)

# true [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
# predictions [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]

# confusion matrix - the permuted labels destroy the predictions - yey
cnfmtrx = confusion_matrix(y1_true,y1_pred)
print(confusion_matrix(y1_true,y1_pred)) # rows - true, col - prediction
# [[0 8]
#  [6 2]]

cm = confusion_matrix(y1_true, y1_pred, labels=model.classes_) # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
disp01 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp01.plot()
# plt.savefig('confusion matrix permuted labels.pdf')
plt.show() #


# datasetF1 = pandas.read_csv('myo6tot_0SD.csv') # , header = None) # all columns of controls
#
# print(datasetF1.head()) # [5 rows x 33 columns]
# #    Chicken Band  TRT  bw 0d  bw 7d  ...  fcr 3wk  fcr 4wk  fcr 5wk  WB level
# # 0          2382  Con   40.4  191.7  ...     1.03     1.39     1.78         0
# # 1          2340  Con   38.5  181.2  ...     1.34     1.40     1.46         0
# # 2          2398  Con   41.2  156.1  ...     1.24     1.33     1.38         2
# # 3          2261  Con   43.5  209.0  ...     1.17     1.57     1.65         0
# # 4          2309  Con   43.7  211.3  ...     1.16     1.35     1.41         3
#
# arr0 = datasetF1.iloc[:, 2:].values
#
#
#
# # print(statistics.mean(arr0[0])) # 1.3419411222669196e-16
# # arr0lst = arr0.tolist()
# # gp1 = [1] * 12
# # gp2 = [3] * 12
# # gp3 = [4] * 12
# #
# # gpnames0 = gp1 + gp2 + gp3
# # mrkrsgrps1 = arr0lst + [gpnames0]
# # mrkrsgrps = np.array(mrkrsgrps1)
# # mrkrsforRF = mrkrsgrps.transpose() # array, each line is an animal, each column a feature
# # print(len(mrkrsforRF), len(mrkrsforRF[0]), mrkrsforRF[0, -1] ) # 36 213 1.0
# #
# X = arr0[:, 0:-1]
# Y = arr0[:, -1]
# classifier = RandomForestClassifier(random_state=0)
#
# cvout = LeaveOneOut()
# # # enumerate splits
# y_true, y_pred = list(), list()
# for train_ix, test_ix in cvout.split(X):
#     # print(test_ix)
#     # split data
#     X_train, X_test = X[train_ix, :], X[test_ix, :]
#     y_train, y_test = Y[train_ix], Y[test_ix]
#     # fit model
#     model = RandomForestClassifier(random_state=1)
#     model.fit(X_train, y_train)
#     # evaluate model
#     yhat = model.predict(X_test)
#     # store
#     y_true.append(y_test[0])
#     y_pred.append(yhat[0])
#
# # calculate accuracy
# acc = accuracy_score(y_true, y_pred)
# print('Accuracy: %.3f' % acc)
# # Accuracy: 0.594
# print('true, predictions', y_true, y_pred)
# # True [0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 2.0, 3.0, 3.0, 1.0, 2.0, 3.0, 2.0, 0.0, 3.0, 1.0, 0.0, 0.0, 3.0, 0.0, 2.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 3.0, 2.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 2.0, 1.0, 3.0, 3.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0, 3.0, 3.0, 3.0, 2.0, 1.0, 3.0]
# # Prediction [3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 3.0, 3.0, 3.0, 0.0, 3.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 3.0, 2.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 3.0, 3.0, 3.0, 3.0, 2.0, 3.0]
# #
#
# # confusion matrix
# cnfmtrx = confusion_matrix(y_true,y_pred)
# print(confusion_matrix(y_true,y_pred)) # rows - true, col - prediction
# # [[42  0  1  6]
# #  [ 5  0  1  2]
# #  [ 8  0  3  7]
# #  [ 8  0  3 15]]
#
# cm = confusion_matrix(y_true, y_pred, labels=model.classes_) # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
# disp0 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
# disp0.plot()
# # plt.savefig('confusion matrix control t1t3t4.pdf')
# plt.show() #
#
# datasetF2 = pandas.read_csv('myo6tot_0SD_reddeath.csv') # , header = None) # all columns of controls
#
# print(datasetF1.head()) #
# #   Chicken Band  TRT  bw 0d  bw 7d  ...  fcr 3wk  fcr 4wk  fcr 5wk  WB level
# # 0          2382  Con   40.4  191.7  ...     1.03     1.39     1.78         0
# # 1          2340  Con   38.5  181.2  ...     1.34     1.40     1.46         0
# # 2          2398  Con   41.2  156.1  ...     1.24     1.33     1.38         2
# # 3          2261  Con   43.5  209.0  ...     1.17     1.57     1.65         0
# # 4          2309  Con   43.7  211.3  ...     1.16     1.35     1.41         3
#
#
# arr01 = datasetF2.iloc[:, 2:].values
#
#
#
# # print(statistics.mean(arr0[0])) # 1.3419411222669196e-16
# # arr0lst = arr0.tolist()
# # gp1 = [1] * 12
# # gp2 = [3] * 12
# # gp3 = [4] * 12
# #
# # gpnames0 = gp1 + gp2 + gp3
# # mrkrsgrps1 = arr0lst + [gpnames0]
# # mrkrsgrps = np.array(mrkrsgrps1)
# # mrkrsforRF = mrkrsgrps.transpose() # array, each line is an animal, each column a feature
# # print(len(mrkrsforRF), len(mrkrsforRF[0]), mrkrsforRF[0, -1] ) # 36 213 1.0
# #
# X = arr01[:, 0:-1]
# Y = arr01[:, -1]
# classifier = RandomForestClassifier(random_state=0)
#
# cvout = LeaveOneOut()
# # # enumerate splits
# y_true, y_pred = list(), list()
# for train_ix, test_ix in cvout.split(X):
#     # print(test_ix)
#     # split data
#     X_train, X_test = X[train_ix, :], X[test_ix, :]
#     y_train, y_test = Y[train_ix], Y[test_ix]
#     # fit model
#     model = RandomForestClassifier(random_state=1)
#     model.fit(X_train, y_train)
#     # evaluate model
#     yhat = model.predict(X_test)
#     # store
#     y_true.append(y_test[0])
#     y_pred.append(yhat[0])
#
# # calculate accuracy
# acc = accuracy_score(y_true, y_pred)
# print('Accuracy: %.3f' % acc)
# # Accuracy: 0.495
# print('true, predictions', y_true, y_pred)
# # true [0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 2.0, 3.0, 3.0, 1.0, 2.0, 3.0, 2.0, 0.0, 3.0, 1.0, 0.0, 0.0, 3.0, 0.0, 2.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 3.0, 2.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 2.0, 1.0, 3.0, 3.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0, 3.0, 3.0, 3.0, 2.0, 1.0, 3.0]
# # predictions [3.0, 3.0, 3.0, 0.0, 3.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 3.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 2.0, 0.0, 2.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 3.0, 3.0, 0.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 2.0, 3.0, 3.0]
#
# # confusion matrix
# cnfmtrx = confusion_matrix(y_true,y_pred)
# print(confusion_matrix(y_true,y_pred)) # rows - true, col - prediction
# # [[35  0  1 13]
# #  [ 4  0  1  3]
# #  [10  0  3  5]
# #  [12  0  2 12]]
#
# cm = confusion_matrix(y_true, y_pred, labels=model.classes_) # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
# disp0 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
# disp0.plot()
# # plt.savefig('confusion matrix control t1t3t4.pdf')
# plt.show() #











































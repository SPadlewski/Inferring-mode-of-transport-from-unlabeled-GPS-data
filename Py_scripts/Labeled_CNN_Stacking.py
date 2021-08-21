#!/usr/bin/env python
# coding: utf-8

# In[12]:


# stacked generalization with linear meta model on blobs dataset
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
from numpy import dstack
from keras.utils import np_utils
import time
import numpy as np
import random
import pickle
from sklearn.model_selection import train_test_split


# In[14]:


# load models from file
def load_all_models(n_models):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = 'models/model_CNN_' + str(i + 1) + '.h5'
		# load model from file
		model = load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models

# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX_M,inputX_G):
    stackX = None
    for i in range(len(members)):
        model=members[i]
        if i<7:
            # make prediction
            yhat = model.predict(inputX_M, verbose=0)
            # stack predictions into [rows, members, probabilities]
            if stackX is None:
                stackX = yhat
            else:
                stackX = dstack((stackX, yhat))
        else:
            # make prediction
            yhat = model.predict(inputX_G, verbose=0)
            # stack predictions into [rows, members, probabilities]
            if stackX is None:
                stackX = yhat
            else:
                stackX = dstack((stackX, yhat))            
                
    # flatten predictions to [rows, members x probabilities]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    return stackX

# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
    # create dataset using ensemble
    inputX_M=inputX[:, :, :, :4]
    inputX_G=inputX[:, :, :, 4:]
    stackedX = stacked_dataset(members, inputX_M,inputX_G)
    # fit standalone model
    model = LogisticRegression(max_iter=6500)
    model.fit(stackedX, inputy)
    return model

# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
    # create dataset using ensemble
    inputX_M=inputX[:, :, :, :4]
    inputX_G=inputX[:, :, :, 4:]
    stackedX = stacked_dataset(members, inputX_M,inputX_G)
    # make a prediction
    yhat = model.predict(stackedX)
    return yhat


start_time = time.clock()
np.random.seed(7)
random.seed(7)

filename = 'Combined Trajectory_Label_Geolife/Revised_KerasData_Smoothing_GIS_new_8.pickle'

with open(filename, mode='rb') as f:
    TotalInput, FinalLabel = pickle.load(f, encoding='latin1')  # Also can use the encoding 'iso-8859-1'

NoClass = len(list(set(np.ndarray.flatten(FinalLabel))))
Threshold = len(TotalInput[0, 0, :, 0])

# Making training and test data: 80% Training, 20% Test
Train_X, Test_X, Train_Y, Test_Y_ori = train_test_split(TotalInput, FinalLabel, test_size=0.20, random_state=7)

Test_M_X=Test_X[:, :, :, :4]
Test_G_X=Test_X[:, :, :, 4:]


# load all models
n_members = 14
members = load_all_models(n_members)
print('Loaded %d models' % len(members))
# evaluate standalone models on test dataset
for i in range(len(members)):
    model=members[i]
    if i <7:
        testy_enc = np_utils.to_categorical(Test_Y_ori, num_classes=NoClass)
        _, acc = model.evaluate(Test_M_X, testy_enc, verbose=0)
        print('Model Accuracy: %.3f' % acc)
    else:
        testy_enc = np_utils.to_categorical(Test_Y_ori, num_classes=NoClass)
        _, acc = model.evaluate(Test_G_X, testy_enc, verbose=0)
        print('Model Accuracy: %.3f' % acc)
# fit stacked model using the ensemble
model = fit_stacked_model(members, Test_X, Test_Y_ori)
# evaluate model on test set
yhat = stacked_prediction(members, model, Test_X)
acc = accuracy_score(Test_Y_ori, yhat)
print('Stacked Test Accuracy: %.3f' % acc)


# In[6]:


from sklearn.metrics import confusion_matrix
import time
start_time = time.clock()
NoClass=5
counter = 0
for i in range(len(yhat)):
    if yhat[i] == Test_Y_ori[i]:
        counter += 1
Accuracy = counter * 1./len(yhat)

ActualPositive = []
for i in range(NoClass):
    AA = np.where(Test_Y_ori == i)[0]
    ActualPositive.append(AA)

PredictedPositive = []
for i in range(NoClass):
    AA = np.where(yhat == i)[0]
    PredictedPositive.append(AA)

TruePositive = []
FalsePositive = []
for i in range(NoClass):
    AA = []
    BB = []
    for j in PredictedPositive[i]:
        if yhat[j] == Test_Y_ori[j]:
            AA.append(j)
        else:
            BB.append(j)
    TruePositive.append(AA)
    FalsePositive.append(BB)
Precision = []
Recall = []
for i in range(NoClass):
    Precision.append(len(TruePositive[i]) * 1./len(PredictedPositive[i]))
    Recall.append(len(TruePositive[i]) * 1./len(ActualPositive[i]))

ConfusionM = confusion_matrix(list(Test_Y_ori), yhat, labels=[0, 1, 2, 3, 4])

print('Confusion Matrix: ', ConfusionM)
print("Recall", Recall)
print('precision', Precision)
print(time.clock() - start_time, "seconds")


# In[53]:


sum(Recall)/len(Recall)


# In[54]:


sum(Precision)/len(Precision)


# In[55]:


F_score=[]
for i in range(len(Precision)):
    f=(2*Precision[i]*Recall[i])/(Precision[i]+Recall[i])
    F_score.append(f)


# In[56]:


sum(F_score)/len(F_score)


#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import pickle
from keras.optimizers import Adam
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import time
import random
from keras.utils import np_utils
#import tensorflow.compat.v1 as tf
#tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
#sess = print(tf.Session(config=tf.ConfigProto(log_device_placement=True)))


# In[6]:


start_time = time.clock()
np.random.seed(7)
random.seed(7)
os.chdir(r'/home/jupyter/Combined Trajectory_Label_Geolife')
filename = '../Combined Trajectory_Label_Geolife/Revised_KerasData_Smoothing.pickle'
with open(filename, mode='rb') as f:
    TotalInput, FinalLabel = pickle.load(f, encoding='latin1')  # Also can use the encoding 'iso-8859-1'

NoClass = len(list(set(np.ndarray.flatten(FinalLabel))))
Threshold = len(TotalInput[0, 0, :, 0])

# Making training and test data: 80% Training, 20% Test
Train_X, Test_X, Train_Y, Test_Y_ori = train_test_split(TotalInput, FinalLabel, test_size=0.20, random_state=7)

Train_Y = np_utils.to_categorical(Train_Y, num_classes=NoClass)
Test_Y = np_utils.to_categorical(Test_Y_ori, num_classes=NoClass)


# In[4]:


# Model and Compile
model = Sequential()

model.add(Conv2D(32, (1, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(1, Threshold, 4)))
model.add(Conv2D(32, (1, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 2)))

model.add(Conv2D(64, (1, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(64, (1, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 2)))

model.add(Conv2D(128, (1, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(Conv2D(128, (1, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Dropout(.5))

model.add(Flatten())
A = model.output_shape
model.add(Dense(int(A[1] * 1/4.), activation='relu'))
model.add(Dropout(.5))

model.add(Dense(NoClass, activation='softmax'))

optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Ensemble configuration
score = []
Pred = []
for i in range(7):
    np.random.seed((i + 5) * 2)#1->5
    Number = np.random.choice(len(Train_X), size=len(Train_X), replace=True, p=None)
    Ens_Train_X = np.zeros((len(Train_X), 1, Threshold, 4), dtype=float)
    Ens_Train_Y = np.zeros((len(Train_Y), NoClass), dtype=float)
    counter = 0
    for j in Number:
        Ens_Train_X[counter, :, :, :] = Train_X[j, :, :, :]
        Ens_Train_Y[counter, :] = Train_Y[j, :]
        counter += 1

    #Ens_Train_Y = np_utils.to_categorical(Ens_Train_Y, num_classes=NoClass)
    
    #Ens_Train_X, testX, Ens_Train_Y, testy = train_test_split(Train_X, Train_Y, test_size=0.0001)
    
    model.fit(Ens_Train_X, Ens_Train_Y, epochs=62, batch_size=64, shuffle=False)
    score.append(model.evaluate(Test_X, Test_Y, batch_size=64))
    Pred.append(model.predict(Test_X, batch_size=64))
    # save model
    filename = '../models/model_CNN_' + str(i + 1) + '.h5'
    model.save(filename)


# In[8]:


# Saving the test and training score for varying number of epochs.
#with open('../models/ensemble_Motion_2_.pickle', 'wb') as f:
#    pickle.dump([score,Pred], f)


# In[18]:


with open('../models/ensemble_GIS_69.pickle', 'rb') as f:
    score,Pred = pickle.load(f, encoding='latin1')


# In[19]:


score


# In[20]:


# Calculating the accuracy, precision, recall.
CombinedPred = np.mean(Pred, axis=0)
Pred_Label = np.argmax(CombinedPred, axis=1)

counter = 0
for i in range(len(Pred_Label)):
    if Pred_Label[i] == Test_Y_ori[i]:
        counter += 1
EnsembleAccuracy = counter * 1./len(Pred_Label)

PredictedPositive = []
for i in range(NoClass):
    AA = np.where(Pred_Label == i)[0]
    PredictedPositive.append(AA)

ActualPositive = []
for i in range(NoClass):
    AA = np.where(Test_Y_ori == i)[0]
    ActualPositive.append(AA)

TruePositive = []
FalsePositive = []
for i in range(NoClass):
    AA = []
    BB = []
    for j in PredictedPositive[i]:
        if Pred_Label[j] == Test_Y_ori[j]:
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

ConfusionM = confusion_matrix(list(Test_Y_ori), Pred_Label, labels=range(NoClass))

print(score)
print('Ensemble Accuracy: ', EnsembleAccuracy)
print('Confusion Matrix: ', ConfusionM)
print("Recall", Recall)
print('Precision', Precision)
print(time.clock() - start_time, "seconds")


# In[21]:


print(sum(Recall)/len(Recall))
print(sum(Precision)/len(Precision))


# In[23]:


sum(F_score)/len(F_score)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[138]:


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


# In[3]:


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

    
filename = 'Data/Revised_KerasData_Smoothing_8_60_final.pickle'

with open(filename, mode='rb') as f:
    TotalInput_UN, FinalStage_UN = pickle.load(f, encoding='latin1')  # Also can use the encoding 'iso-8859-1'
    
    
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


# In[4]:


#loading segments for classification 
filename = 'Data/Revised_KerasData_Smoothing_8_40_final_segmentID.pickle'

with open(filename, mode='rb') as f:
    TotalInput_UN, FinalStage_UN = pickle.load(f, encoding='latin1')  # Also can use the encoding 'iso-8859-1'


# In[6]:


#selecting validation sample
Train_X, Test_X, Train_Y, Test_Y_ori = train_test_split(TotalInput_UN, FinalStage_UN, test_size=0.2, random_state=7)


# In[7]:


get_ipython().run_cell_magic('time', '', '#classifiaction\nyhat = stacked_prediction(members, model, Test_X)')


# In[9]:


#validation sample size
len(yhat)


# In[10]:


#loading segments GPS points for validation and plotting
filename = 'Data/Proximity+200_Unlabeled_40_final.pickle'
with open(filename, 'rb') as f:
    Bus_All_Segment,SegmentID,Rail_All_Segment, Traffic_All_Segment, Stage, Data_All_Segment, SegmentNumber = pickle.load(f, encoding='latin1')


# In[11]:


get_ipython().run_cell_magic('time', '', "# assigning segments ID to predicted modes\nimport pandas as pd\nsegments_index=[i for i,x in enumerate(SegmentID) if x in Test_Y_ori]\n\ndf_pred = pd.DataFrame(columns = ['lat','lon','timestamp','segmentID','stageID','pred_mode'])\nfor i in range(len(segments_index)):\n    df = pd.DataFrame(Data_All_Segment[segments_index[i]])\n    df=df.rename(columns={0:'lat',1:'lon',2:'timestamp'})\n    df['segmentID']=Test_Y_ori[i]\n    df['stageID']=Stage[segments_index[i]]\n    df['pred_mode']=yhat[i]   \n    df_pred=pd.concat([df_pred,df]).reset_index(drop=True)")


# In[13]:


df_pred


# In[139]:


#saving data frame with predicted modes
#df_pred.to_pickle('Data/segments_40_1_percent_predictions.pkl')
df_pred = pd.read_pickle('Data/segments_40_1_percent_predictions.pkl')


# ### Validation process

# In[276]:


# selecting segments with modes 0,1,2,3 or 4 (walk,bike,bus,driving or train)
df_pred[df_pred.pred_mode==1].segmentID.unique()


# In[339]:


#selecing a segment for validation
import skmob
import folium

df_pred[df_pred.segmentID==5740]


# In[341]:


#plotting selected segment for validation

data= skmob.TrajDataFrame(
df_pred[df_pred.segmentID==5740],
latitude= "lat",
longitude="lon",
datetime='timestamp',
parameters= {1:'segmentID'})

#plotting
data.plot_trajectory(zoom=12, weight=10, opacity=0.9, tiles='OpenStreetMap'
                     ,start_end_markers=True,dashArray='5, 5')


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
from geopy.distance import geodesic # used to be vincenty
import os
import math
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer
from sklearn.neighbors import BallTree
import pandas as pd


os.chdir(r'/home/jupyter/Combined Trajectory_Label_Geolife')


# In[2]:


A = math.degrees(-math.pi)
# Change the current working directory to the location of 'Combined Trajectory_Label_Geolife' folder.
#filename = '../Combined Trajectory_Label_Geolife/Revised_Trajectory_Label_Array.pickle'
with open('Revised_Trajectory_Label_Array.pickle', 'rb') as f:
    Trajectory_Label_Array = pickle.load(f)


# In[4]:


len(Trajectory_Label_Array)
NoPoints=0
for i in range(len(Trajectory_Label_Array)):
    NoPoints+=len(Trajectory_Label_Array[i])
print(NoPoints)    


# In[5]:


# Identify the Speed and Acceleration limit
SpeedLimit = {0: 7, 1: 12, 2: 120./3.6, 3: 180./3.6, 4: 120/3.6}
# Online sources for Acc: walk: 1.5 Train 1.15, bus. 1.25 (.2), bike: 2.6, train:1.5
AccLimit = {0: 3, 1: 3, 2: 2, 3: 10, 4: 3}
# Choose based on figure visualization for JerkP:{0: 4, 1: 4, 2: 4, 3: 11, 4: 6}
JerkLimitP = {0: 40, 1: 40, 2: 40, 3: 110, 4: 60}
# Choose based on figure visualization for JerkN:{0: -4, 1: -4, 2: -2.5, 3: -11, 4: -4}
JerkLimitN = {0: -40, 1: -40, 2: -200.5, 3: -110, 4: -40}


# In[6]:


# Total_Instance_InSequence checks the number of GPS points for each instance in all users
Total_Instance_InSequence = []
# Total_Motion_Instance: each element is an array include the four channels for each instance
Total_Motion_Instance = []
# Save the 4 channels for each user separately
Total_RelativeDistance = []
Total_Speed = []
Total_Acceleration = []
Total_Jerk = []
Total_BearingRate = []
Total_Label = []
Total_InstanceNumber = []
Total_Outlier = []
Total_Descriptive_Stat = []
Total_Delta_Time = []
Total_Velocity_Change = []
Total_BusLine = []
Total_Railway = []
Total_Traffic = []


# In[7]:


Trajectory_Label_Array[1][68253]


# In[8]:


for z in range(len(Trajectory_Label_Array)):
    Data = Trajectory_Label_Array[z]
    for i in range(len(Data) - 1):
        a=Data[i, 0]
        if a> 70:
            Data[i, 0]=Data[i, 0]-360
            print(z,i)
        elif a< 0:
            print(z,i)


# In[10]:


get_ipython().run_cell_magic('time', '', '#Szymon\'s Version\n# Count the number of times that NoOfOutlier happens\nNoOfOutlier = 0\nfor z in range(len(Trajectory_Label_Array)):# iterate of users 69 ##len(Trajectory_Label_Array)\n    print(z)\n    Descriptive_Stat = []\n    Data = Trajectory_Label_Array[z]\n    if len(Data) == 0:\n        continue\n\n    Shape = np.shape(Trajectory_Label_Array[z])\n    # InstanceNumber: Break a user\'s trajectory to instances. Count number of GPS points for each instance\n    delta_time = []\n    tempSpeed = []# list with speed between two consecutive points\n    for i in range(len(Data) - 1):\n        delta_time.append((Data[i+1, 2] - Data[i, 2]) * 24. * 60*60)\n        if delta_time[i] == 0:\n            # Prevent to generate infinite speed. So use a very short time = 0.1 seconds.\n            delta_time[i] = 0.1\n        A = (Data[i, 0], Data[i, 1])# starting position\n        B = (Data[i + 1, 0], Data[i + 1, 1])# end position\n        tempSpeed.append(geodesic(A, B).meters/delta_time[i]) #speed between two consecutive points\n    # Since there is no data for the last point, we assume the delta_time as the average time in the user guide\n    # (i.e., 3 sec) and speed as tempSpeed equal to last time so far.\n    delta_time.append(3)\n    tempSpeed.append(tempSpeed[len(tempSpeed) - 1])\n\n    # InstanceNumber: indicate the length of each instance\n    InstanceNumber = []\n    # Label: For each created instance, we need only one mode to be assigned to.\n    # Remove the instance with less than 10 GPS points. Break the whole user\'s trajectory into trips with min_trip\n    # Also break the instance with more than threshold GPS points into more instances\n    Data_All_Instance = []  # Each of its element is a list that shows the data for each instance (lat, long, time)\n    Label = []\n    min_trip_time = 20 * 60  # 20 minutes equal to 1200 seconds\n    threshold = 200  # fixed of number of GPS points for each instance\n    i = 0\n    while i <= (len(Data) - 1):\n        No = 0\n        ModeType = Data[i, 3]\n        Counter = 0\n        # index: save the instance indices when an instance is being created and concatenate all in the remove\n        index = []\n        # First, we always have an instance with one GPS point.\n        while i <= (len(Data) - 1) and Data[i, 3] == ModeType and Counter < threshold:\n            if delta_time[i] <= min_trip_time:\n                Counter += 1\n                index.append(i)\n                i += 1\n            else:\n                Counter += 1\n                index.append(i)\n                i += 1\n                break\n\n        if Counter >= 10:  # Remove all instances that have less than 10 GPS points# I\n            InstanceNumber.append(Counter)\n            Data_For_Instance = [Data[i, 0:3] for i in index]\n            Data_For_Instance = np.array(Data_For_Instance, dtype=float)\n            Data_All_Instance.append(Data_For_Instance)            \n            \n            Label.append(ModeType)\n\n    if len(InstanceNumber) == 0:\n        continue\n\n    Label = [int(i) for i in Label] #label to int\n    \n    #creating empty list for every pair of points\n    RelativeDistance = [[] for _ in range(len(InstanceNumber))] \n    Speed = [[] for _ in range(len(InstanceNumber))]\n    Acceleration = [[] for _ in range(len(InstanceNumber))]\n    Jerk = [[] for _ in range(len(InstanceNumber))]\n    Bearing = [[] for _ in range(len(InstanceNumber))]\n    BearingRate = [[] for _ in range(len(InstanceNumber))]\n    Delta_Time = [[] for _ in range(len(InstanceNumber))]\n    Velocity_Change = [[] for _ in range(len(InstanceNumber))]\n    User_outlier = []\n    \n    ###### Create channels for every instance (k) of the current user (k) ######\n    for k in range(len(InstanceNumber)):\n        Data = Data_All_Instance[k] # a list of points in a instance\n        # Temp_RD, Temp_SP are temporary relative distance and speed before checking for their length\n        Temp_Speed = []\n        Temp_RD = []\n        outlier = []\n        for i in range(len(Data) - 1):\n            A = (Data[i, 0], Data[i, 1])\n            B = (Data[i+1, 0], Data[i+1, 1])\n            Temp_RD.append(geodesic(A, B).meters)\n            Delta_Time[k].append((Data[i + 1, 2] - Data[i, 2]) * 24. * 3600 + 1)  # Add one second to prevent zero time\n            S = Temp_RD[i] / Delta_Time[k][i]\n            if S > SpeedLimit[Label[k]] or S < 0:\n                outlier.append(i)\n            Temp_Speed.append(S)\n            \n            #Calculating Bearing\n            y = math.sin(math.radians(Data[i+1, 1]) - math.radians(Data[i, 1])) * math.radians(math.cos(Data[i+1, 0]))\n            x = math.radians(math.cos(Data[i, 0])) * math.radians(math.sin(Data[i+1, 0])) - \\\n                math.radians(math.sin(Data[i, 0])) * math.radians(math.cos(Data[i+1, 0])) \\\n                * math.radians(math.cos(Data[i+1, 1]) - math.radians(Data[i, 1]))\n            # Convert radian from -pi to pi to [0, 360] degree\n            b = (math.atan2(y, x) * 180. / math.pi + 360) % 360\n            Bearing[k].append(b)\n\n        # End of operation of relative distance, speed, and bearing for one instance\n        \n        # Now remove all outliers (exceeding max speed) in the current instance\n        Temp_Speed = [i for j, i in enumerate(Temp_Speed) if j not in outlier]\n        if len(Temp_Speed) < 10:\n            InstanceNumber[k] = 0\n            NoOfOutlier += 1\n            continue\n        Speed[k] = Temp_Speed\n        Speed[k].append(Speed[k][-1])\n\n        # Now remove all outlier instances, where their speed exceeds the max speed.\n        # Then, remove their corresponding points from other channels.\n        RelativeDistance[k] = Temp_RD\n        RelativeDistance[k] = [i for j, i in enumerate(RelativeDistance[k]) if j not in outlier]\n        RelativeDistance[k].append(RelativeDistance[k][-1])\n        Bearing[k] = [i for j, i in enumerate(Bearing[k]) if j not in outlier]\n        Bearing[k].append(Bearing[k][-1])\n        Delta_Time[k] = [i for j, i in enumerate(Delta_Time[k]) if j not in outlier]\n\n        InstanceNumber[k] = InstanceNumber[k] - len(outlier) #decrease the number of points in the instance \n\n        # Now remove all outlier instances, where their acceleration exceeds the max acceleration.\n        # Then, remove their corresponding points from other channels.\n        Temp_ACC = []\n        outlier = []\n        for i in range(len(Speed[k]) - 1):\n            DeltaSpeed = Speed[k][i+1] - Speed[k][i]\n            ACC = DeltaSpeed/Delta_Time[k][i]\n            if abs(ACC) > AccLimit[Label[k]]:\n                outlier.append(i)\n            Temp_ACC.append(ACC)\n\n        Temp_ACC = [i for j, i in enumerate(Temp_ACC) if j not in outlier]\n        if len(Temp_ACC) < 10:\n            InstanceNumber[k] = 0\n            NoOfOutlier += 1\n            continue\n        Acceleration[k] = Temp_ACC\n        Acceleration[k].append(Acceleration[k][-1])\n        Speed[k] = [i for j, i in enumerate(Speed[k]) if j not in outlier]\n        RelativeDistance[k] = [i for j, i in enumerate(RelativeDistance[k]) if j not in outlier]\n        Bearing[k] = [i for j, i in enumerate(Bearing[k]) if j not in outlier]\n        Delta_Time[k] = [i for j, i in enumerate(Delta_Time[k]) if j not in outlier]\n\n        InstanceNumber[k] = InstanceNumber[k] - len(outlier)\n\n        # Now remove all outlier instances, where their jerk exceeds the max speed.\n        # Then, remove their corresponding points from other channels.\n\n        Temp_J = []\n        outlier = []\n        for i in range(len(Acceleration[k]) - 1):\n            Diff = Acceleration[k][i+1] - Acceleration[k][i]\n            J = Diff/Delta_Time[k][i]\n            Temp_J.append(J)\n\n        Temp_J = [i for j, i in enumerate(Temp_J) if j not in outlier]\n        if len(Temp_J) < 10:\n            InstanceNumber[k] = 0\n            NoOfOutlier += 1\n            continue\n\n        Jerk[k] = Temp_J\n        Jerk[k].append(Jerk[k][-1])\n        Speed[k] = [i for j, i in enumerate(Speed[k]) if j not in outlier]\n        Acceleration[k] = [i for j, i in enumerate(Acceleration[k]) if j not in outlier]\n        RelativeDistance[k] = [i for j, i in enumerate(RelativeDistance[k]) if j not in outlier]\n        Bearing[k] = [i for j, i in enumerate(Bearing[k]) if j not in outlier]\n        Delta_Time[k] = [i for j, i in enumerate(Delta_Time[k]) if j not in outlier]\n\n        InstanceNumber[k] = InstanceNumber[k] - len(outlier)\n        # End of Jerk outlier detection.\n\n        # Compute Breating Rate from Bearing, and Velocity change from Speed\n        for i in range(len(Bearing[k]) - 1):\n            Diff = abs(Bearing[k][i+1] - Bearing[k][i])\n            BearingRate[k].append(Diff)\n        BearingRate[k].append(BearingRate[k][-1])\n\n        for i in range(len(Speed[k]) - 1):\n            Diff = abs(Speed[k][i+1] - Speed[k][i])\n            if Speed[k][i] != 0:\n                Velocity_Change[k].append(Diff/Speed[k][i])\n            else:\n                Velocity_Change[k].append(1)\n        Velocity_Change[k].append(Velocity_Change[k][-1])\n        \n               # Now we apply the smoothing filter on each instance:\n        def savitzky_golay(y, window_size, order, deriv=0, rate=1):\n            r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.\n            The Savitzky-Golay filter removes high frequency noise from data.\n            It has the advantage of preserving the original shape and\n            features of the signal better than other types of filtering\n            approaches, such as moving averages techniques.\n            Parameters\n            ----------\n            y : array_like, shape (N,)\n                the values of the time history of the signal.\n            window_size : int\n                the length of the window. Must be an odd integer number.\n            order : int\n                the order of the polynomial used in the filtering.\n                Must be less then `window_size` - 1.\n            deriv: int\n                the order of the derivative to compute (default = 0 means only smoothing)\n            Returns\n            -------\n            ys : ndarray, shape (N)\n                the smoothed signal (or it\'s n-th derivative).\n            Notes\n            -----\n            The Savitzky-Golay is a type of low-pass filter, particularly\n            suited for smoothing noisy data. The main idea behind this\n            approach is to make for each point a least-square fit with a\n            polynomial of high order over a odd-sized window centered at\n            the point.\n            Examples\n            --------\n            t = np.linspace(-4, 4, 500)\n            y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)\n            ysg = savitzky_golay(y, window_size=31, order=4)\n            import matplotlib.pyplot as plt\n            plt.plot(t, y, label=\'Noisy signal\')\n            plt.plot(t, np.exp(-t**2), \'k\', lw=1.5, label=\'Original signal\')\n            plt.plot(t, ysg, \'r\', label=\'Filtered signal\')\n            plt.legend()\n            plt.show()\n            References\n            ----------\n            .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of\n               Data by Simplified Least Squares Procedures. Analytical\n               Chemistry, 1964, 36 (8), pp 1627-1639.\n            .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing\n               W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery\n               Cambridge University Press ISBN-13: 9780521880688\n            """\n            import numpy as np\n            from math import factorial\n\n            try:\n                window_size = np.abs(np.int(window_size))\n                order = np.abs(np.int(order))\n            except ValueError:\n                raise ValueError("window_size and order have to be of type int")\n            if window_size % 2 != 1 or window_size < 1:\n                raise TypeError("window_size size must be a positive odd number")\n            if window_size < order + 2:\n                raise TypeError("window_size is too small for the polynomials order")\n            order_range = range(order + 1)\n            half_window = (window_size - 1) // 2\n            # precompute coefficients\n            b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])\n            m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)\n            # pad the signal at the extremes with\n            # values taken from the signal itself\n            firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])\n            lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])\n            y = np.concatenate((firstvals, y, lastvals))\n            return np.convolve(m[::-1], y, mode=\'valid\')\n\n        # Smoothing process\n        RelativeDistance[k] = savitzky_golay(np.array(RelativeDistance[k]), 9, 3)\n        Speed[k] = savitzky_golay(np.array(Speed[k]), 9, 3)\n        Acceleration[k] = savitzky_golay(np.array(Acceleration[k]), 9, 3)\n        Jerk[k] = savitzky_golay(np.array(Jerk[k]), 9, 3)\n        BearingRate[k] = savitzky_golay(np.array(BearingRate[k]), 9, 3)\n        \n    Total_RelativeDistance.append(RelativeDistance)\n    Total_Speed.append(Speed)\n    Total_Acceleration.append(Acceleration)\n    Total_Jerk.append(Jerk)\n    Total_BearingRate.append(BearingRate)\n    Total_Delta_Time.append(Delta_Time)\n    Total_Velocity_Change.append(Velocity_Change)\n    Total_Label.append(Label)\n    Total_InstanceNumber.append(InstanceNumber)\n    Total_Outlier.append(User_outlier)\n    Total_Instance_InSequence = Total_Instance_InSequence + InstanceNumber')


# In[11]:


with open('Revised_InstanceCreation+Smoothing+WHOLE_COUNTRY.pickle', 'wb') as f:
    pickle.dump([Total_RelativeDistance, Total_Speed, Total_Acceleration, Total_Jerk, Total_BearingRate, Total_Label,
                 Total_InstanceNumber, Total_Instance_InSequence, Total_Delta_Time, Total_Velocity_Change,Total_BusLine,Total_Railway,Total_Traffic], f)


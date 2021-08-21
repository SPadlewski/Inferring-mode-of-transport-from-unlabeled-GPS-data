#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime,time
from datetime import datetime
import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
import skmob
import gc
import sys
import math
from shapely.geometry import Point
from skmob.preprocessing import filtering
from skmob.preprocessing import detection
from geopy.distance import geodesic
from sklearn.neighbors import BallTree
from scipy.signal import savgol_filter


# In[2]:


filename = 'Data/Proximity+200_Unlabeled_60_final.pickle'
with open(filename, 'rb') as f:
    Bus_All_Segment,SegmentID ,Rail_All_Segment, Traffic_All_Segment, Stage, Data_All_Segment, SegmentNumber = pickle.load(f, encoding='latin1')


# # Calculating Motion channels

# In[3]:


# Total_Segment_InSequence checks the number of GPS points for each instance in all Stages
Total_Segment_InSequence = []
# Save the 4 channels for each user separately
Total_RelativeDistance = []
Total_Speed = []
Total_Acceleration = []
Total_Jerk = []
Total_BearingRate = []
Total_Stage = []
Total_SegmentNumber = []
Total_Outlier = []
Total_Descriptive_Stat = []
Total_Delta_Time = []
Total_Velocity_Change = []
Total_BusLine = []
Total_Railway = []
Total_Traffic = []


# In[4]:


get_ipython().run_cell_magic('time', '', '#Calculating Channels for segemnts\n# Count the number of times that NoOfOutlier happens\nNoOfOutlier = 0\n\nStage = [int(i) for i in Stage] #Stage to int\n\n    \n#creating empty list for every pair of points\nRelativeDistance = [[] for _ in range(len(SegmentNumber))] \nSpeed = [[] for _ in range(len(SegmentNumber))]\nAcceleration = [[] for _ in range(len(SegmentNumber))]\nJerk = [[] for _ in range(len(SegmentNumber))]\nBearing = [[] for _ in range(len(SegmentNumber))]\nBearingRate = [[] for _ in range(len(SegmentNumber))]\nDelta_Time = [[] for _ in range(len(SegmentNumber))]\nVelocity_Change = [[] for _ in range(len(SegmentNumber))]\nUser_outlier = []\n    \n###### Create channels for every Segment (k) \nfor k in range(len(SegmentNumber)):\n    Data = Data_All_Segment[k] # a list of points in a Segment\n    # Temp_RD, Temp_SP are temporary relative distance and speed before checking for their length\n    Temp_Speed = []\n    Temp_RD = []\n    outlier = []\n    for i in range(len(Data) - 1):\n        A = (Data[i, 0], Data[i, 1])\n        B = (Data[i+1, 0], Data[i+1, 1])\n        Temp_RD.append(geodesic(A, B).meters)\n        Delta_Time[k].append((Data[i + 1, 2] - Data[i, 2]) * 24. * 3600 + 1)  # Add one second to prevent zero time\n        S = Temp_RD[i] / Delta_Time[k][i]\n        if S > 62.5 or S < 0: # max speed of the fastest rail in the UK\n            outlier.append(i)\n        Temp_Speed.append(S)\n            \n        #Calculating Bearing\n        y = math.sin(math.radians(Data[i+1, 1]) - math.radians(Data[i, 1])) * math.radians(math.cos(Data[i+1, 0]))\n        x = math.radians(math.cos(Data[i, 0])) * math.radians(math.sin(Data[i+1, 0])) - \\\n        math.radians(math.sin(Data[i, 0])) * math.radians(math.cos(Data[i+1, 0])) \\\n            * math.radians(math.cos(Data[i+1, 1]) - math.radians(Data[i, 1]))\n        # Convert radian from -pi to pi to [0, 360] degree\n        b = (math.atan2(y, x) * 180. / math.pi + 360) % 360\n        Bearing[k].append(b)\n\n        \n    # End of operation of relative distance, speed, and bearing for one Segment\n        \n    # Now remove all outliers (exceeding max speed) in the current Segment\n    Temp_Speed = [i for j, i in enumerate(Temp_Speed) if j not in outlier]        \n    if len(Temp_Speed) < 10:\n        SegmentNumber[k] = 0\n        NoOfOutlier += 1\n        continue\n    Speed[k] = Temp_Speed\n    Speed[k].append(Speed[k][-1])\n\n    # Now remove all outlier Segments, where their speed exceeds the max speed.\n    # Then, remove their corresponding points from other channels.\n    RelativeDistance[k] = Temp_RD\n    RelativeDistance[k] = [i for j, i in enumerate(RelativeDistance[k]) if j not in outlier]\n    RelativeDistance[k].append(RelativeDistance[k][-1])\n    Bearing[k] = [i for j, i in enumerate(Bearing[k]) if j not in outlier]\n    Bearing[k].append(Bearing[k][-1])\n    Delta_Time[k] = [i for j, i in enumerate(Delta_Time[k]) if j not in outlier]\n    \n    SegmentNumber[k] = SegmentNumber[k] - len(outlier) #decrease the number of points in the Segment \n\n    # Now remove all outlier Segments, where their acceleration exceeds the max acceleration.\n    # Then, remove their corresponding points from other channels.\n    Temp_ACC = []\n    outlier = []\n    for i in range(len(Speed[k]) - 1):\n        DeltaSpeed = Speed[k][i+1] - Speed[k][i]\n        ACC = DeltaSpeed/Delta_Time[k][i]\n        if abs(ACC) > 10:\n            outlier.append(i)\n        Temp_ACC.append(ACC)\n\n    Temp_ACC = [i for j, i in enumerate(Temp_ACC) if j not in outlier]\n    if len(Temp_ACC) < 10:\n        SegmentNumber[k] = 0\n        NoOfOutlier += 1\n        continue\n    Acceleration[k] = Temp_ACC\n    Acceleration[k].append(Acceleration[k][-1])\n    Speed[k] = [i for j, i in enumerate(Speed[k]) if j not in outlier]\n    RelativeDistance[k] = [i for j, i in enumerate(RelativeDistance[k]) if j not in outlier]\n    Bearing[k] = [i for j, i in enumerate(Bearing[k]) if j not in outlier]\n    Delta_Time[k] = [i for j, i in enumerate(Delta_Time[k]) if j not in outlier]\n\n    SegmentNumber[k] = SegmentNumber[k] - len(outlier)\n\n    # Now remove all outlier Segments, where their jerk exceeds the max speed.\n    # Then, remove their corresponding points from other channels.\n\n    Temp_J = []\n    outlier = []\n    for i in range(len(Acceleration[k]) - 1):\n        Diff = Acceleration[k][i+1] - Acceleration[k][i]\n        J = Diff/Delta_Time[k][i]\n        Temp_J.append(J)\n\n    Temp_J = [i for j, i in enumerate(Temp_J) if j not in outlier]\n    if len(Temp_J) < 10:\n        SegmentNumber[k] = 0\n        NoOfOutlier += 1\n        continue\n\n    Jerk[k] = Temp_J\n    Jerk[k].append(Jerk[k][-1])\n    Speed[k] = [i for j, i in enumerate(Speed[k]) if j not in outlier]\n    Acceleration[k] = [i for j, i in enumerate(Acceleration[k]) if j not in outlier]\n    RelativeDistance[k] = [i for j, i in enumerate(RelativeDistance[k]) if j not in outlier]\n    Bearing[k] = [i for j, i in enumerate(Bearing[k]) if j not in outlier]\n    Delta_Time[k] = [i for j, i in enumerate(Delta_Time[k]) if j not in outlier]\n\n    SegmentNumber[k] = SegmentNumber[k] - len(outlier)\n    # End of Jerk outlier detection.\n\n    # Compute Breating Rate from Bearing, and Velocity change from Speed\n    for i in range(len(Bearing[k]) - 1):\n        Diff = abs(Bearing[k][i+1] - Bearing[k][i])\n        BearingRate[k].append(Diff)\n    BearingRate[k].append(BearingRate[k][-1])\n\n    for i in range(len(Speed[k]) - 1):\n        Diff = abs(Speed[k][i+1] - Speed[k][i])\n        if Speed[k][i] != 0:\n            Velocity_Change[k].append(Diff/Speed[k][i])\n        else:\n            Velocity_Change[k].append(1)\n    Velocity_Change[k].append(Velocity_Change[k][-1])\n        \n        \n    # Now we apply the smoothing filter on each Segment:\n    def savitzky_golay(y, window_size, order, deriv=0, rate=1):\n        r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.\n        The Savitzky-Golay filter removes high frequency noise from data.\n        It has the advantage of preserving the original shape and\n        features of the signal better than other types of filtering\n        approaches, such as moving averages techniques.\n        Parameters\n        ----------\n        y : array_like, shape (N,)\n            the values of the time history of the signal.\n            window_size : int\n            the length of the window. Must be an odd integer number.\n        order : int\n            the order of the polynomial used in the filtering.\n            Must be less then `window_size` - 1.\n        deriv: int\n            the order of the derivative to compute (default = 0 means only smoothing)\n        Returns\n        -------\n        ys : ndarray, shape (N)\n            the smoothed signal (or it\'s n-th derivative).\n        Notes\n        -----\n        The Savitzky-Golay is a type of low-pass filter, particularly\n        suited for smoothing noisy data. The main idea behind this\n        approach is to make for each point a least-square fit with a\n        polynomial of high order over a odd-sized window centered at\n        the point.\n        Examples\n        --------\n        t = np.linspace(-4, 4, 500)\n        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)\n        ysg = savitzky_golay(y, window_size=31, order=4)\n        import matplotlib.pyplot as plt\n        plt.plot(t, y, label=\'Noisy signal\')\n        plt.plot(t, np.exp(-t**2), \'k\', lw=1.5, label=\'Original signal\')\n        plt.plot(t, ysg, \'r\', label=\'Filtered signal\')\n        plt.legend()\n        plt.show()\n        References\n        ----------\n        .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of\n        Data by Simplified Least Squares Procedures. Analytical\n           Chemistry, 1964, 36 (8), pp 1627-1639.\n        .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing\n        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery\n           Cambridge University Press ISBN-13: 9780521880688\n        """\n        import numpy as np\n        from math import factorial\n\n        try:\n            window_size = np.abs(np.int(window_size))\n            order = np.abs(np.int(order))\n        except ValueError:\n            raise ValueError("window_size and order have to be of type int")\n        if window_size % 2 != 1 or window_size < 1:\n            raise TypeError("window_size size must be a positive odd number")\n        if window_size < order + 2:\n            raise TypeError("window_size is too small for the polynomials order")\n        order_range = range(order + 1)\n        half_window = (window_size - 1) // 2\n        # precompute coefficients\n        b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])\n        m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)\n        # pad the signal at the extremes with\n        # values taken from the signal itself\n        firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])\n        lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])\n        y = np.concatenate((firstvals, y, lastvals))\n        return np.convolve(m[::-1], y, mode=\'valid\')\n\n    # Smoothing process\n    #RelativeDistance[k] = savitzky_golay(np.array(RelativeDistance[k]), 9, 3)\n    #Speed[k] = savitzky_golay(np.array(Speed[k]), 9, 3)\n    #Acceleration[k] = savitzky_golay(np.array(Acceleration[k]), 9, 3)\n    #Jerk[k] = savitzky_golay(np.array(Jerk[k]), 9, 3)\n    #BearingRate[k] = savitzky_golay(np.array(BearingRate[k]), 9, 3)\n        \nTotal_RelativeDistance.append(RelativeDistance)\nTotal_Speed.append(Speed)\nTotal_Acceleration.append(Acceleration)\nTotal_Jerk.append(Jerk)\nTotal_BearingRate.append(BearingRate)\nTotal_Delta_Time.append(Delta_Time)\nTotal_Velocity_Change.append(Velocity_Change)\nTotal_Stage.append(Stage)\nTotal_SegmentNumber.append(SegmentNumber)\nTotal_Outlier.append(User_outlier)\nTotal_Segment_InSequence = Total_Segment_InSequence + SegmentNumber')


# In[5]:


#saving results
with open('Data/Revised_Unlabeled_60_NoSmoothing_final.pickle', 'wb') as f:
    pickle.dump([Total_RelativeDistance, Total_Speed, Total_Acceleration, Total_Jerk, Total_BearingRate, Total_Stage,
                 Total_SegmentNumber, Total_Segment_InSequence, Total_Delta_Time, Total_Velocity_Change,Total_BusLine,Total_Railway,Total_Traffic], f)


# In[ ]:





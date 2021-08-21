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


# In[56]:


#loading data
df_dis = pd.read_pickle('Data/stages_40_final.pkl')


# In[57]:


df_dis#stageID_final.nunique()#stageID_final#54430 and 147028


# In[58]:


#changing datetime format
df_dis['bench']=np.datetime64('1899-12-30')
df_dis['bench']=df_dis['bench'].dt.tz_localize('UTC')

z=df_dis['datetime']
x=(df_dis['datetime']-df_dis['bench'])
df_dis['timestamp']=x.dt.days + z.dt.hour/24 + z.dt.minute / (24. * 60.) + z.dt.second / (24. * 3600.)
df_dis=df_dis.reset_index()


# In[59]:


#Selecting needed columns
df_dis=df_dis[['lat','lon','timestamp','stageID_final']]


# In[60]:


#converting df_dis to numpy array
df_dis_array=df_dis.to_numpy()


# # Calculating proximity channels 

# In[61]:


railStop_gdf_en = gpd.read_file("GIS/EN_gis_osm_transport_free_1.shp",crs='EPSG:4326')
busStop_gdf_en=railStop_gdf_en[(railStop_gdf_en.fclass=="bus_station")|(railStop_gdf_en.fclass=="bus_stop")]
railStop_gdf_en=railStop_gdf_en[(railStop_gdf_en.fclass=="railway_station")]


railStop_gdf_sc = gpd.read_file("GIS/SC_gis_osm_transport_free_1.shp",crs='EPSG:4326')
busStop_gdf_sc=railStop_gdf_sc[(railStop_gdf_sc.fclass=="bus_station")|(railStop_gdf_sc.fclass=="bus_stop")]
railStop_gdf_sc=railStop_gdf_sc[(railStop_gdf_sc.fclass=="railway_station")]

railStop_gdf_wa = gpd.read_file("GIS/WA_gis_osm_transport_free_1.shp",crs='EPSG:4326')
busStop_gdf_wa=railStop_gdf_wa[(railStop_gdf_wa.fclass=="bus_station")|(railStop_gdf_wa.fclass=="bus_stop")]
railStop_gdf_wa=railStop_gdf_wa[(railStop_gdf_wa.fclass=="railway_station")]


railStop_gdf_ni = gpd.read_file("GIS/NI_rail.shp",crs='EPSG:4326')
busStop_gdf_ni = gpd.read_file("GIS/NI_bus.shp",crs='EPSG:4326')


railStop_gdf = railStop_gdf_en.append(railStop_gdf_wa)
railStop_gdf = railStop_gdf.append(railStop_gdf_sc)
railStop_gdf = railStop_gdf.append(railStop_gdf_ni)

busStop_gdf = busStop_gdf_en.append(busStop_gdf_wa)
busStop_gdf = busStop_gdf.append(busStop_gdf_sc)
busStop_gdf = busStop_gdf.append(busStop_gdf_ni)


# In[62]:


trafficStop_gdf_wa = gpd.read_file("GIS/WA_gis_osm_traffic_free_1.shp",crs='EPSG:4326')
trafficStop_gdf_wa=trafficStop_gdf_wa[(trafficStop_gdf_wa.fclass=="crossing")|(trafficStop_gdf_wa.fclass=="motorway_junction")|(trafficStop_gdf_wa.fclass=="traffic_signals")]

trafficStop_gdf_sc = gpd.read_file("GIS/SC_gis_osm_traffic_free_1.shp",crs='EPSG:4326')
trafficStop_gdf_sc=trafficStop_gdf_sc[(trafficStop_gdf_sc.fclass=="crossing")|(trafficStop_gdf_sc.fclass=="motorway_junction")|(trafficStop_gdf_sc.fclass=="traffic_signals")]

trafficStop_gdf_en = gpd.read_file("GIS/EN_gis_osm_traffic_free_1.shp",crs='EPSG:4326')
trafficStop_gdf_en=trafficStop_gdf_en[(trafficStop_gdf_en.fclass=="crossing")|(trafficStop_gdf_en.fclass=="motorway_junction")|(trafficStop_gdf_en.fclass=="traffic_signals")]

trafficStop_gdf_ni = gpd.read_file("GIS/NI_traffic.shp",crs='EPSG:4326')

trafficStop_gdf = trafficStop_gdf_en.append(trafficStop_gdf_wa)
trafficStop_gdf = trafficStop_gdf.append(trafficStop_gdf_sc)
trafficStop_gdf = trafficStop_gdf.append(trafficStop_gdf_ni)


# In[63]:


print(trafficStop_gdf.shape)
print(railStop_gdf.shape)
print(busStop_gdf.shape)


# In[64]:


(186753, 5)
(3499, 5)
(251717, 5)


# In[65]:


def get_nearest(src_points, candidates, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points"""

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
    closest = indices[0]
    closest_dist = distances[0]

    # Return indices and distances
    return (closest, closest_dist)


def nearest_neighbor(left_gdf, right_gdf, return_dist=False):
    """
    For each point in left_gdf, find closest point in right GeoDataFrame and return them.
    
    NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
    """
    
    left_geom_col = left_gdf.geometry.name
    right_geom_col = right_gdf.geometry.name
    
    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf.copy().reset_index(drop=True)
    
    # Parse coordinates from points and insert them into a numpy array as RADIANS
    # Notice: should be in Lat/Lon format 
    left_radians = np.array(left_gdf[left_geom_col].apply(lambda geom: (geom.y * np.pi / 180, geom.x * np.pi / 180)).to_list())
    right_radians = np.array(right[right_geom_col].apply(lambda geom: (geom.y * np.pi / 180, geom.x * np.pi / 180)).to_list())
    
    # Find the nearest points
    # -----------------------
    # closest ==> index in right_gdf that corresponds to the closest point
    # dist ==> distance between the nearest neighbors (in meters)
    
    closest, dist = get_nearest(src_points=left_radians, candidates=right_radians)

    # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
    closest_points = right.loc[closest]
    
    # Ensure that the index corresponds the one in left_gdf
    closest_points = closest_points.reset_index(drop=True)
    
    # Add distance if requested 
    if return_dist:
        # Convert to meters from radians
        earth_radius = 6371000  # meters
        closest_points['distance'] = dist * earth_radius
        
    return closest_points


# In[66]:


get_ipython().run_cell_magic('time', '', "#32min or 1h 17min\nRailStop_dis=[]\nTrafficStop_dis=[]\nBusStop_dis=[]\n\nData = df_dis_array\n\narr1lat = np.ravel(Data[:,0])\narr1lon = np.ravel(Data[:,1])\ndf1 = pd.DataFrame({'lat':arr1lat, 'lon':arr1lon})\ndf1['coords'] = list(zip( df1['lon'],df1['lat']))\ndf1['coords'] = df1['coords'].apply(Point)\ngdf1 = gpd.GeoDataFrame(df1, geometry='coords')\n    \nclosest_stops_bus = nearest_neighbor(gdf1 ,busStop_gdf, return_dist=True)\nclosest_stops_traffic = nearest_neighbor(gdf1 ,trafficStop_gdf, return_dist=True)\nclosest_stops_rail = nearest_neighbor(gdf1 ,railStop_gdf, return_dist=True)\n    \nbus_dis = closest_stops_bus['distance'].values\ntraffic_dis = closest_stops_traffic['distance'].values\nrail_dis = closest_stops_rail['distance'].values\n    \nBusStop_dis.append(bus_dis)\nTrafficStop_dis.append(traffic_dis)\nRailStop_dis.append(rail_dis)")


# In[67]:


#saving results
with open('Data/Proximity_Unlabeled_40_final.pickle', 'wb') as f:
    pickle.dump([BusStop_dis,TrafficStop_dis,RailStop_dis], f)


# In[68]:


filename = 'Data/Proximity_Unlabeled_40_final.pickle'
with open(filename, 'rb') as f:
    BusStop_dis,TrafficStop_dis,RailStop_dis = pickle.load(f, encoding='latin1')


# In[69]:


get_ipython().run_cell_magic('time', '', '#dividing stages into segments with 200 points\nData=df_dis_array\n# SegmentNumber: indicate the length of each Segment\nSegmentNumber = []\n# Stage: For each created segment, we need only one mode to be assigned to.\n# Remove the segments with less than 10 GPS points. \n# Also break the stages with more than threshold GPS points into more segment\nBusData=BusStop_dis[0]\nRailData=RailStop_dis[0]\nTrafficData=TrafficStop_dis[0]\n\nBus_All_Segment=[]\nRail_All_Segment=[]\nTraffic_All_Segment=[]\nStage = []\nData_All_Segment = []  # Each of its element is a list that shows the data for each segment (lat, long, time)\nthreshold = 200  # fixed of number of GPS points for each segment\ni = 0\nwhile i <= (len(Data) - 1):\n    No = 0\n    StageID = Data[i, 3]\n    Counter = 0\n    # index: save the segment indices when an Segment is being created and concatenate all in the remove\n    index = []\n    # First, we always have an segment with one GPS point.\n    while i <= (len(Data) - 1) and Data[i, 3] == StageID and Counter < threshold:\n        Counter += 1\n        index.append(i)\n        i += 1\n\n    if Counter >= 10:  # Remove all segment that have less than 10 GPS points\n        SegmentNumber.append(Counter)\n        Data_For_Segment = [Data[i, 0:3] for i in index]#[0:3]\n        Data_For_Segment = np.array(Data_For_Segment, dtype=float)\n        Data_All_Segment.append(Data_For_Segment)            \n        \n        Bus_For_Segment=[BusData[i] for i in index]\n        Bus_All_Segment.append(Bus_For_Segment)\n        \n        Rail_For_Segment=[RailData[i] for i in index]\n        Rail_All_Segment.append(Rail_For_Segment)\n        \n        Traffic_For_Segment=[TrafficData[i] for i in index]\n        Traffic_All_Segment.append(Traffic_For_Segment)\n        Stage.append(StageID)')


# In[70]:


len(Stage)


# In[71]:


SegmentID = [*range(0, len(Stage))]


# In[72]:


#saving results
with open('Data/Proximity+200_Unlabeled_40_final.pickle', 'wb') as f:
    pickle.dump([Bus_All_Segment,SegmentID, Rail_All_Segment, Traffic_All_Segment, Stage, Data_All_Segment, SegmentNumber], f)


# In[2]:


filename = 'Data/Proximity+200_Unlabeled_40_final.pickle'
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


get_ipython().run_cell_magic('time', '', '#Calculating Channels for segemnts\n# Count the number of times that NoOfOutlier happens\nNoOfOutlier = 0\n\nStage = [int(i) for i in Stage] #Stage to int\n\n    \n#creating empty list for every pair of points\nRelativeDistance = [[] for _ in range(len(SegmentNumber))] \nSpeed = [[] for _ in range(len(SegmentNumber))]\nAcceleration = [[] for _ in range(len(SegmentNumber))]\nJerk = [[] for _ in range(len(SegmentNumber))]\nBearing = [[] for _ in range(len(SegmentNumber))]\nBearingRate = [[] for _ in range(len(SegmentNumber))]\nDelta_Time = [[] for _ in range(len(SegmentNumber))]\nVelocity_Change = [[] for _ in range(len(SegmentNumber))]\nUser_outlier = []\n    \n###### Create channels for every Segment (k) \nfor k in range(len(SegmentNumber)):\n    Data = Data_All_Segment[k] # a list of points in a Segment\n    # Temp_RD, Temp_SP are temporary relative distance and speed before checking for their length\n    Temp_Speed = []\n    Temp_RD = []\n    outlier = []\n    for i in range(len(Data) - 1):\n        A = (Data[i, 0], Data[i, 1])\n        B = (Data[i+1, 0], Data[i+1, 1])\n        Temp_RD.append(geodesic(A, B).meters)\n        Delta_Time[k].append((Data[i + 1, 2] - Data[i, 2]) * 24. * 3600 + 1)  # Add one second to prevent zero time\n        S = Temp_RD[i] / Delta_Time[k][i]\n        if S > 62.5 or S < 0: # max speed of the fastest rail in the UK\n            outlier.append(i)\n        Temp_Speed.append(S)\n            \n        #Calculating Bearing\n        y = math.sin(math.radians(Data[i+1, 1]) - math.radians(Data[i, 1])) * math.radians(math.cos(Data[i+1, 0]))\n        x = math.radians(math.cos(Data[i, 0])) * math.radians(math.sin(Data[i+1, 0])) - \\\n        math.radians(math.sin(Data[i, 0])) * math.radians(math.cos(Data[i+1, 0])) \\\n            * math.radians(math.cos(Data[i+1, 1]) - math.radians(Data[i, 1]))\n        # Convert radian from -pi to pi to [0, 360] degree\n        b = (math.atan2(y, x) * 180. / math.pi + 360) % 360\n        Bearing[k].append(b)\n\n        \n    # End of operation of relative distance, speed, and bearing for one Segment\n        \n    # Now remove all outliers (exceeding max speed) in the current Segment\n    Temp_Speed = [i for j, i in enumerate(Temp_Speed) if j not in outlier]        \n    if len(Temp_Speed) < 10:\n        SegmentNumber[k] = 0\n        NoOfOutlier += 1\n        continue\n    Speed[k] = Temp_Speed\n    Speed[k].append(Speed[k][-1])\n\n    # Now remove all outlier Segments, where their speed exceeds the max speed.\n    # Then, remove their corresponding points from other channels.\n    Bus_All_Segment[k] = [i for j, i in enumerate(Bus_All_Segment[k]) if j not in outlier]\n    Rail_All_Segment[k] = [i for j, i in enumerate(Rail_All_Segment[k]) if j not in outlier]\n    Traffic_All_Segment[k] = [i for j, i in enumerate(Traffic_All_Segment[k]) if j not in outlier]                               \n    RelativeDistance[k] = Temp_RD\n    RelativeDistance[k] = [i for j, i in enumerate(RelativeDistance[k]) if j not in outlier]\n    RelativeDistance[k].append(RelativeDistance[k][-1])\n    Bearing[k] = [i for j, i in enumerate(Bearing[k]) if j not in outlier]\n    Bearing[k].append(Bearing[k][-1])\n    Delta_Time[k] = [i for j, i in enumerate(Delta_Time[k]) if j not in outlier]\n    \n    SegmentNumber[k] = SegmentNumber[k] - len(outlier) #decrease the number of points in the Segment \n\n    # Now remove all outlier Segments, where their acceleration exceeds the max acceleration.\n    # Then, remove their corresponding points from other channels.\n    Temp_ACC = []\n    outlier = []\n    for i in range(len(Speed[k]) - 1):\n        DeltaSpeed = Speed[k][i+1] - Speed[k][i]\n        ACC = DeltaSpeed/Delta_Time[k][i]\n        if abs(ACC) > 10:\n            outlier.append(i)\n        Temp_ACC.append(ACC)\n\n    Temp_ACC = [i for j, i in enumerate(Temp_ACC) if j not in outlier]\n    if len(Temp_ACC) < 10:\n        SegmentNumber[k] = 0\n        NoOfOutlier += 1\n        continue\n    Acceleration[k] = Temp_ACC\n    Acceleration[k].append(Acceleration[k][-1])\n    Bus_All_Segment[k] = [i for j, i in enumerate(Bus_All_Segment[k]) if j not in outlier]\n    Rail_All_Segment[k] = [i for j, i in enumerate(Rail_All_Segment[k]) if j not in outlier]\n    Traffic_All_Segment[k] = [i for j, i in enumerate(Traffic_All_Segment[k]) if j not in outlier]                        \n    Speed[k] = [i for j, i in enumerate(Speed[k]) if j not in outlier]\n    RelativeDistance[k] = [i for j, i in enumerate(RelativeDistance[k]) if j not in outlier]\n    Bearing[k] = [i for j, i in enumerate(Bearing[k]) if j not in outlier]\n    Delta_Time[k] = [i for j, i in enumerate(Delta_Time[k]) if j not in outlier]\n\n    SegmentNumber[k] = SegmentNumber[k] - len(outlier)\n\n    # Now remove all outlier Segments, where their jerk exceeds the max speed.\n    # Then, remove their corresponding points from other channels.\n\n    Temp_J = []\n    outlier = []\n    for i in range(len(Acceleration[k]) - 1):\n        Diff = Acceleration[k][i+1] - Acceleration[k][i]\n        J = Diff/Delta_Time[k][i]\n        Temp_J.append(J)\n\n    Temp_J = [i for j, i in enumerate(Temp_J) if j not in outlier]\n    if len(Temp_J) < 10:\n        SegmentNumber[k] = 0\n        NoOfOutlier += 1\n        continue\n\n    Jerk[k] = Temp_J\n    Jerk[k].append(Jerk[k][-1])\n    Bus_All_Segment[k] = [i for j, i in enumerate(Bus_All_Segment[k]) if j not in outlier]\n    Rail_All_Segment[k] = [i for j, i in enumerate(Rail_All_Segment[k]) if j not in outlier]\n    Traffic_All_Segment[k] = [i for j, i in enumerate(Traffic_All_Segment[k]) if j not in outlier] \n    Speed[k] = [i for j, i in enumerate(Speed[k]) if j not in outlier]\n    Acceleration[k] = [i for j, i in enumerate(Acceleration[k]) if j not in outlier]\n    RelativeDistance[k] = [i for j, i in enumerate(RelativeDistance[k]) if j not in outlier]\n    Bearing[k] = [i for j, i in enumerate(Bearing[k]) if j not in outlier]\n    Delta_Time[k] = [i for j, i in enumerate(Delta_Time[k]) if j not in outlier]\n\n    SegmentNumber[k] = SegmentNumber[k] - len(outlier)\n    # End of Jerk outlier detection.\n\n    # Compute Breating Rate from Bearing, and Velocity change from Speed\n    for i in range(len(Bearing[k]) - 1):\n        Diff = abs(Bearing[k][i+1] - Bearing[k][i])\n        BearingRate[k].append(Diff)\n    BearingRate[k].append(BearingRate[k][-1])\n\n    for i in range(len(Speed[k]) - 1):\n        Diff = abs(Speed[k][i+1] - Speed[k][i])\n        if Speed[k][i] != 0:\n            Velocity_Change[k].append(Diff/Speed[k][i])\n        else:\n            Velocity_Change[k].append(1)\n    Velocity_Change[k].append(Velocity_Change[k][-1])\n        \n        \n    # Now we apply the smoothing filter on each Segment:\n    def savitzky_golay(y, window_size, order, deriv=0, rate=1):\n        r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.\n        The Savitzky-Golay filter removes high frequency noise from data.\n        It has the advantage of preserving the original shape and\n        features of the signal better than other types of filtering\n        approaches, such as moving averages techniques.\n        Parameters\n        ----------\n        y : array_like, shape (N,)\n            the values of the time history of the signal.\n            window_size : int\n            the length of the window. Must be an odd integer number.\n        order : int\n            the order of the polynomial used in the filtering.\n            Must be less then `window_size` - 1.\n        deriv: int\n            the order of the derivative to compute (default = 0 means only smoothing)\n        Returns\n        -------\n        ys : ndarray, shape (N)\n            the smoothed signal (or it\'s n-th derivative).\n        Notes\n        -----\n        The Savitzky-Golay is a type of low-pass filter, particularly\n        suited for smoothing noisy data. The main idea behind this\n        approach is to make for each point a least-square fit with a\n        polynomial of high order over a odd-sized window centered at\n        the point.\n        Examples\n        --------\n        t = np.linspace(-4, 4, 500)\n        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)\n        ysg = savitzky_golay(y, window_size=31, order=4)\n        import matplotlib.pyplot as plt\n        plt.plot(t, y, label=\'Noisy signal\')\n        plt.plot(t, np.exp(-t**2), \'k\', lw=1.5, label=\'Original signal\')\n        plt.plot(t, ysg, \'r\', label=\'Filtered signal\')\n        plt.legend()\n        plt.show()\n        References\n        ----------\n        .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of\n        Data by Simplified Least Squares Procedures. Analytical\n           Chemistry, 1964, 36 (8), pp 1627-1639.\n        .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing\n        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery\n           Cambridge University Press ISBN-13: 9780521880688\n        """\n        import numpy as np\n        from math import factorial\n\n        try:\n            window_size = np.abs(np.int(window_size))\n            order = np.abs(np.int(order))\n        except ValueError:\n            raise ValueError("window_size and order have to be of type int")\n        if window_size % 2 != 1 or window_size < 1:\n            raise TypeError("window_size size must be a positive odd number")\n        if window_size < order + 2:\n            raise TypeError("window_size is too small for the polynomials order")\n        order_range = range(order + 1)\n        half_window = (window_size - 1) // 2\n        # precompute coefficients\n        b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])\n        m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)\n        # pad the signal at the extremes with\n        # values taken from the signal itself\n        firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])\n        lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])\n        y = np.concatenate((firstvals, y, lastvals))\n        return np.convolve(m[::-1], y, mode=\'valid\')\n\n    # Smoothing process\n    RelativeDistance[k] = savitzky_golay(np.array(RelativeDistance[k]), 9, 3)\n    Speed[k] = savitzky_golay(np.array(Speed[k]), 9, 3)\n    Acceleration[k] = savitzky_golay(np.array(Acceleration[k]), 9, 3)\n    Jerk[k] = savitzky_golay(np.array(Jerk[k]), 9, 3)\n    BearingRate[k] = savitzky_golay(np.array(BearingRate[k]), 9, 3)\n    Bus_All_Segment[k]= savitzky_golay(np.array(Bus_All_Segment[k]), 9, 3)\n    Rail_All_Segment[k]= savitzky_golay(np.array(Rail_All_Segment[k]), 9, 3)\n    Traffic_All_Segment[k]= savitzky_golay(np.array(Traffic_All_Segment[k]), 9, 3)\n        \nTotal_RelativeDistance.append(RelativeDistance)\nTotal_Speed.append(Speed)\nTotal_Acceleration.append(Acceleration)\nTotal_Jerk.append(Jerk)\nTotal_BearingRate.append(BearingRate)\nTotal_BusLine.append(Bus_All_Segment)\nTotal_Railway.append(Rail_All_Segment)\nTotal_Traffic.append(Traffic_All_Segment)                       \nTotal_Delta_Time.append(Delta_Time)\nTotal_Velocity_Change.append(Velocity_Change)\nTotal_Stage.append(Stage)\nTotal_SegmentNumber.append(SegmentNumber)\nTotal_Outlier.append(User_outlier)\nTotal_Segment_InSequence = Total_Segment_InSequence + SegmentNumber')


# In[5]:


#saving results
with open('Data/Revised_Unlabeled_40_final.pickle', 'wb') as f:
    pickle.dump([Total_RelativeDistance, Total_Speed, Total_Acceleration, Total_Jerk, Total_BearingRate, Total_Stage,
                 Total_SegmentNumber, Total_Segment_InSequence, Total_Delta_Time, Total_Velocity_Change,Total_BusLine,Total_Railway,Total_Traffic], f)


# In[ ]:


# Unifying segment size


# In[6]:


filename = 'Data/Revised_Unlabeled_40_final.pickle'
# Each of the following variables contain multiple lists, where each list belongs to a user
with open(filename, 'rb') as f:
    Total_RelativeDistance, Total_Speed, Total_Acceleration, Total_Jerk, Total_BearingRate, Total_Stage,    Total_SegmentNumber, Total_Segment_InSequence, Total_Delta_Time, Total_Velocity_Change,Total_BusLine,Total_Railway,Total_Traffic = pickle.load(f, encoding='latin1')


# In[7]:


Total_SegmentID =[[int(i) for i in SegmentID]]


# In[8]:


len(Total_SegmentID[0])


# In[9]:


# Create the data in the Keras form
# Threshold: Is the max of number of GPS point in an Segment
#Padding Segments to 200 threshold 
Threshold = 200
Zero_Segment = [i for i, item in enumerate(Total_Segment_InSequence) if item == 0]
Number_of_Segment = len(Total_Segment_InSequence) - len(Zero_Segment)
TotalInput = np.zeros((Number_of_Segment, 1, Threshold, 8), dtype=float)
FinalStage = np.zeros((Number_of_Segment, 1), dtype=int)

counter = 0
for k in range(len(Total_SegmentNumber)):
    # Create Keras shape with 8 channels for each user
    #  There are 8 channels(in order: RelativeDistance, Speed, Acceleration, BearingRate)
    RD = Total_RelativeDistance[k]
    SP = Total_Speed[k]
    AC = Total_Acceleration[k]
    J = Total_Jerk[k]
    BR = Total_BearingRate[k]
    LA = Total_SegmentID[k]#Total_Stage
    BS=Total_BusLine[k]
    RL=Total_Railway[k]
    TR=Total_Traffic[k]

    
    # IN: the Segments and number of GPS points in each Segment for each user k
    IN = Total_SegmentNumber[k]

    for i in range(len(IN)):
        end = IN[i]
        if end == 0 or sum(RD[i]) == 0:
            continue
        TotalInput[counter, 0, 0:end, 0] = SP[i]
        TotalInput[counter, 0, 0:end, 1] = AC[i]
        TotalInput[counter, 0, 0:end, 2] = J[i]
        TotalInput[counter, 0, 0:end, 3] = BR[i]
        TotalInput[counter, 0, 0:end, 4] = BS[i]           
        TotalInput[counter, 0, 0:end, 5] = RL[i]
        TotalInput[counter, 0, 0:end, 6] = TR[i]
        TotalInput[counter, 0, 0:end, 7] = RD[i]
        
        FinalStage[counter, 0] = LA[i]
        counter += 1

TotalInput = TotalInput[:counter, :, :, :]
FinalStage = FinalStage[:counter, 0]


# In[10]:


with open('Data/Revised_KerasData_Smoothing_8_40_final_segmentID.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([TotalInput, FinalStage], f)


# In[ ]:





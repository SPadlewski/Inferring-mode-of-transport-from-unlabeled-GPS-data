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


#32min or 1h 17min
RailStop_dis=[]
TrafficStop_dis=[]
BusStop_dis=[]

Data = df_dis_array

arr1lat = np.ravel(Data[:,0])
arr1lon = np.ravel(Data[:,1])
df1 = pd.DataFrame({'lat':arr1lat, 'lon':arr1lon})
df1['coords'] = list(zip( df1['lon'],df1['lat']))
df1['coords'] = df1['coords'].apply(Point)
gdf1 = gpd.GeoDataFrame(df1, geometry='coords')
    
closest_stops_bus = nearest_neighbor(gdf1 ,busStop_gdf, return_dist=True)
closest_stops_traffic = nearest_neighbor(gdf1 ,trafficStop_gdf, return_dist=True)
closest_stops_rail = nearest_neighbor(gdf1 ,railStop_gdf, return_dist=True)
    
bus_dis = closest_stops_bus['distance'].values
traffic_dis = closest_stops_traffic['distance'].values
rail_dis = closest_stops_rail['distance'].values
    
BusStop_dis.append(bus_dis)
TrafficStop_dis.append(traffic_dis)
RailStop_dis.append(rail_dis)


# In[67]:


#saving results
with open('Data/Proximity_Unlabeled_40_final.pickle', 'wb') as f:
    pickle.dump([BusStop_dis,TrafficStop_dis,RailStop_dis], f)


# In[68]:


filename = 'Data/Proximity_Unlabeled_40_final.pickle'
with open(filename, 'rb') as f:
    BusStop_dis,TrafficStop_dis,RailStop_dis = pickle.load(f, encoding='latin1')


# In[69]:


#dividing stages into segments with 200 points
Data=df_dis_array
# SegmentNumber: indicate the length of each Segment
SegmentNumber = []
# Stage: For each created segment, we need only one mode to be assigned to.
# Remove the segments with less than 10 GPS points. 
# Also break the stages with more than threshold GPS points into more segment
BusData=BusStop_dis[0]
RailData=RailStop_dis[0]
TrafficData=TrafficStop_dis[0]

Bus_All_Segment=[]
Rail_All_Segment=[]
Traffic_All_Segment=[]
Stage = []
Data_All_Segment = []  # Each of its element is a list that shows the data for each segment (lat, long, time)
threshold = 200  # fixed of number of GPS points for each segment
i = 0
while i <= (len(Data) - 1):
    No = 0
    StageID = Data[i, 3]
    Counter = 0
    # index: save the segment indices when an Segment is being created and concatenate all in the remove
    index = []
    # First, we always have an segment with one GPS point.
    while i <= (len(Data) - 1) and Data[i, 3] == StageID and Counter < threshold:
        Counter += 1
        index.append(i)
        i += 1

    if Counter >= 10:  # Remove all segment that have less than 10 GPS points
        SegmentNumber.append(Counter)
        Data_For_Segment = [Data[i, 0:3] for i in index]#[0:3]
        Data_For_Segment = np.array(Data_For_Segment, dtype=float)
        Data_All_Segment.append(Data_For_Segment)            
        
        Bus_For_Segment=[BusData[i] for i in index]
        Bus_All_Segment.append(Bus_For_Segment)
        
        Rail_For_Segment=[RailData[i] for i in index]
        Rail_All_Segment.append(Rail_For_Segment)
        
        Traffic_For_Segment=[TrafficData[i] for i in index]
        Traffic_All_Segment.append(Traffic_For_Segment)
        Stage.append(StageID)


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


#Calculating Channels for segemnts
# Count the number of times that NoOfOutlier happens
NoOfOutlier = 0

Stage = [int(i) for i in Stage] #Stage to int

    
#creating empty list for every pair of points
RelativeDistance = [[] for _ in range(len(SegmentNumber))] 
Speed = [[] for _ in range(len(SegmentNumber))]
Acceleration = [[] for _ in range(len(SegmentNumber))]
Jerk = [[] for _ in range(len(SegmentNumber))]
Bearing = [[] for _ in range(len(SegmentNumber))]
BearingRate = [[] for _ in range(len(SegmentNumber))]
Delta_Time = [[] for _ in range(len(SegmentNumber))]
Velocity_Change = [[] for _ in range(len(SegmentNumber))]
User_outlier = []
    
###### Create channels for every Segment (k) 
for k in range(len(SegmentNumber)):
    Data = Data_All_Segment[k] # a list of points in a Segment
    # Temp_RD, Temp_SP are temporary relative distance and speed before checking for their length
    Temp_Speed = []
    Temp_RD = []
    outlier = []
    for i in range(len(Data) - 1):
        A = (Data[i, 0], Data[i, 1])
        B = (Data[i+1, 0], Data[i+1, 1])
        Temp_RD.append(geodesic(A, B).meters)
        Delta_Time[k].append((Data[i + 1, 2] - Data[i, 2]) * 24. * 3600 + 1)  # Add one second to prevent zero time
        S = Temp_RD[i] / Delta_Time[k][i]
        if S > 62.5 or S < 0: # max speed of the fastest rail in the UK
            outlier.append(i)
        Temp_Speed.append(S)
            
        #Calculating Bearing
        y = math.sin(math.radians(Data[i+1, 1]) - math.radians(Data[i, 1])) * math.radians(math.cos(Data[i+1, 0]))
        x = math.radians(math.cos(Data[i, 0])) * math.radians(math.sin(Data[i+1, 0])) -         math.radians(math.sin(Data[i, 0])) * math.radians(math.cos(Data[i+1, 0]))             * math.radians(math.cos(Data[i+1, 1]) - math.radians(Data[i, 1]))
        # Convert radian from -pi to pi to [0, 360] degree
        b = (math.atan2(y, x) * 180. / math.pi + 360) % 360
        Bearing[k].append(b)

        
    # End of operation of relative distance, speed, and bearing for one Segment
        
    # Now remove all outliers (exceeding max speed) in the current Segment
    Temp_Speed = [i for j, i in enumerate(Temp_Speed) if j not in outlier]        
    if len(Temp_Speed) < 10:
        SegmentNumber[k] = 0
        NoOfOutlier += 1
        continue
    Speed[k] = Temp_Speed
    Speed[k].append(Speed[k][-1])

    # Now remove all outlier Segments, where their speed exceeds the max speed.
    # Then, remove their corresponding points from other channels.
    Bus_All_Segment[k] = [i for j, i in enumerate(Bus_All_Segment[k]) if j not in outlier]
    Rail_All_Segment[k] = [i for j, i in enumerate(Rail_All_Segment[k]) if j not in outlier]
    Traffic_All_Segment[k] = [i for j, i in enumerate(Traffic_All_Segment[k]) if j not in outlier]                               
    RelativeDistance[k] = Temp_RD
    RelativeDistance[k] = [i for j, i in enumerate(RelativeDistance[k]) if j not in outlier]
    RelativeDistance[k].append(RelativeDistance[k][-1])
    Bearing[k] = [i for j, i in enumerate(Bearing[k]) if j not in outlier]
    Bearing[k].append(Bearing[k][-1])
    Delta_Time[k] = [i for j, i in enumerate(Delta_Time[k]) if j not in outlier]
    
    SegmentNumber[k] = SegmentNumber[k] - len(outlier) #decrease the number of points in the Segment 

    # Now remove all outlier Segments, where their acceleration exceeds the max acceleration.
    # Then, remove their corresponding points from other channels.
    Temp_ACC = []
    outlier = []
    for i in range(len(Speed[k]) - 1):
        DeltaSpeed = Speed[k][i+1] - Speed[k][i]
        ACC = DeltaSpeed/Delta_Time[k][i]
        if abs(ACC) > 10:
            outlier.append(i)
        Temp_ACC.append(ACC)

    Temp_ACC = [i for j, i in enumerate(Temp_ACC) if j not in outlier]
    if len(Temp_ACC) < 10:
        SegmentNumber[k] = 0
        NoOfOutlier += 1
        continue
    Acceleration[k] = Temp_ACC
    Acceleration[k].append(Acceleration[k][-1])
    Bus_All_Segment[k] = [i for j, i in enumerate(Bus_All_Segment[k]) if j not in outlier]
    Rail_All_Segment[k] = [i for j, i in enumerate(Rail_All_Segment[k]) if j not in outlier]
    Traffic_All_Segment[k] = [i for j, i in enumerate(Traffic_All_Segment[k]) if j not in outlier]                        
    Speed[k] = [i for j, i in enumerate(Speed[k]) if j not in outlier]
    RelativeDistance[k] = [i for j, i in enumerate(RelativeDistance[k]) if j not in outlier]
    Bearing[k] = [i for j, i in enumerate(Bearing[k]) if j not in outlier]
    Delta_Time[k] = [i for j, i in enumerate(Delta_Time[k]) if j not in outlier]

    SegmentNumber[k] = SegmentNumber[k] - len(outlier)

    # Now remove all outlier Segments, where their jerk exceeds the max speed.
    # Then, remove their corresponding points from other channels.

    Temp_J = []
    outlier = []
    for i in range(len(Acceleration[k]) - 1):
        Diff = Acceleration[k][i+1] - Acceleration[k][i]
        J = Diff/Delta_Time[k][i]
        Temp_J.append(J)

    Temp_J = [i for j, i in enumerate(Temp_J) if j not in outlier]
    if len(Temp_J) < 10:
        SegmentNumber[k] = 0
        NoOfOutlier += 1
        continue

    Jerk[k] = Temp_J
    Jerk[k].append(Jerk[k][-1])
    Bus_All_Segment[k] = [i for j, i in enumerate(Bus_All_Segment[k]) if j not in outlier]
    Rail_All_Segment[k] = [i for j, i in enumerate(Rail_All_Segment[k]) if j not in outlier]
    Traffic_All_Segment[k] = [i for j, i in enumerate(Traffic_All_Segment[k]) if j not in outlier] 
    Speed[k] = [i for j, i in enumerate(Speed[k]) if j not in outlier]
    Acceleration[k] = [i for j, i in enumerate(Acceleration[k]) if j not in outlier]
    RelativeDistance[k] = [i for j, i in enumerate(RelativeDistance[k]) if j not in outlier]
    Bearing[k] = [i for j, i in enumerate(Bearing[k]) if j not in outlier]
    Delta_Time[k] = [i for j, i in enumerate(Delta_Time[k]) if j not in outlier]

    SegmentNumber[k] = SegmentNumber[k] - len(outlier)
    # End of Jerk outlier detection.

    # Compute Breating Rate from Bearing, and Velocity change from Speed
    for i in range(len(Bearing[k]) - 1):
        Diff = abs(Bearing[k][i+1] - Bearing[k][i])
        BearingRate[k].append(Diff)
    BearingRate[k].append(BearingRate[k][-1])

    for i in range(len(Speed[k]) - 1):
        Diff = abs(Speed[k][i+1] - Speed[k][i])
        if Speed[k][i] != 0:
            Velocity_Change[k].append(Diff/Speed[k][i])
        else:
            Velocity_Change[k].append(1)
    Velocity_Change[k].append(Velocity_Change[k][-1])
        
        
    # Now we apply the smoothing filter on each Segment:
    def savitzky_golay(y, window_size, order, deriv=0, rate=1):
        r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.
        Parameters
        ----------
        y : array_like, shape (N,)
            the values of the time history of the signal.
            window_size : int
            the length of the window. Must be an odd integer number.
        order : int
            the order of the polynomial used in the filtering.
            Must be less then `window_size` - 1.
        deriv: int
            the order of the derivative to compute (default = 0 means only smoothing)
        Returns
        -------
        ys : ndarray, shape (N)
            the smoothed signal (or it's n-th derivative).
        Notes
        -----
        The Savitzky-Golay is a type of low-pass filter, particularly
        suited for smoothing noisy data. The main idea behind this
        approach is to make for each point a least-square fit with a
        polynomial of high order over a odd-sized window centered at
        the point.
        Examples
        --------
        t = np.linspace(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()
        References
        ----------
        .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
        Data by Simplified Least Squares Procedures. Analytical
           Chemistry, 1964, 36 (8), pp 1627-1639.
        .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
           Cambridge University Press ISBN-13: 9780521880688
        """
        import numpy as np
        from math import factorial

        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order + 1)
        half_window = (window_size - 1) // 2
        # precompute coefficients
        b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
        m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve(m[::-1], y, mode='valid')

    # Smoothing process
    RelativeDistance[k] = savitzky_golay(np.array(RelativeDistance[k]), 9, 3)
    Speed[k] = savitzky_golay(np.array(Speed[k]), 9, 3)
    Acceleration[k] = savitzky_golay(np.array(Acceleration[k]), 9, 3)
    Jerk[k] = savitzky_golay(np.array(Jerk[k]), 9, 3)
    BearingRate[k] = savitzky_golay(np.array(BearingRate[k]), 9, 3)
    Bus_All_Segment[k]= savitzky_golay(np.array(Bus_All_Segment[k]), 9, 3)
    Rail_All_Segment[k]= savitzky_golay(np.array(Rail_All_Segment[k]), 9, 3)
    Traffic_All_Segment[k]= savitzky_golay(np.array(Traffic_All_Segment[k]), 9, 3)
        
Total_RelativeDistance.append(RelativeDistance)
Total_Speed.append(Speed)
Total_Acceleration.append(Acceleration)
Total_Jerk.append(Jerk)
Total_BearingRate.append(BearingRate)
Total_BusLine.append(Bus_All_Segment)
Total_Railway.append(Rail_All_Segment)
Total_Traffic.append(Traffic_All_Segment)                       
Total_Delta_Time.append(Delta_Time)
Total_Velocity_Change.append(Velocity_Change)
Total_Stage.append(Stage)
Total_SegmentNumber.append(SegmentNumber)
Total_Outlier.append(User_outlier)
Total_Segment_InSequence = Total_Segment_InSequence + SegmentNumber


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





import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#%%
race_data = pd.read_csv('data/lap_times.csv')
print(race_data.head())

## Separate the data from 2018 to 2023 into a new dataframe
race_data_2018_2023 = race_data[(race_data['raceId'] >= 989) & (race_data['raceId'] <= 1110)]

'''RACE DATA INFO'''
print(race_data_2018_2023.info())
print(race_data_2018_2023.notnull().sum())

#%% Dividing data into seasons
year_intervals = {
    2018: (989, 1009),
    2019: (1010, 1030),
    2020: (1031, 1047),
    2021: (1052, 1073),
    2022: (1074, 1096),
    2023: (1098, 1110)
}
# Dictionary to store each season's data
dataframes = {}
# Extracting each unique raceId for the seasons
unique_raceIds = race_data_2018_2023['raceId'].unique()
# Looping through the dictionary to extract the data for each season
for year, (start, end) in year_intervals.items():
    dataframes[year] = race_data_2018_2023[(race_data_2018_2023['raceId'] >= start) & (race_data_2018_2023['raceId'] <= end)]

#%%
df_race_2018 = dataframes[2018]


#%% Histogram to show the distribution of the lap times
fig, axs = plt.subplots(1, 1, figsize=(30, 20))
sns.histplot(df_race_2018['milliseconds'], bins=150)
plt.xticks(np.arange(66000, 200000, 10000))

''' TEXT 
Based on the histograms, we can see the race data is not normally distributed across across the season.
The data seems to be bimodal, which suggests that there are two distinct subgroups within the data.
The indication of distinct subgroups could be due to various factors llike skill, track conditions or strategy..
'''

#%% Further analysis of the data to identify potential outliers.

print(df_race_2018.describe())

'''TEXT
The histogram and the describe functions both show, that there is an outlier at +500000 milliseconds and potientially at around 200000 milliseconds.
'''

df_race_2018 = df_race_2018[(df_race_2018['milliseconds'] <= 200000) & (df_race_2018['milliseconds'] >= 66000)]

fig, axs = plt.subplots(1, 1, figsize=(30, 20))
sns.histplot(df_race_2018['milliseconds'], bins=150)
plt.xticks(np.arange(66000, 200000, 1000))

df_race_2018.describe()

Q1 = df_race_2018['milliseconds'].quantile(0.25)
Q3 = df_race_2018['milliseconds'].quantile(0.75)
IQR = Q3 - Q1
print(Q1, Q3, IQR)
axs.fill_betweenx([0, 1000], Q1, Q3, color='red', alpha=0.3)

#%% Extracting the driver data
driver_race_times = df_race_2018.groupby('driverId')[['driverId', 'raceId', 'milliseconds''']].apply(lambda x: x)
driver_race_times['driverId_cat'] = pd.Categorical(df_race_2018['driverId']).codes

#%% Scatterplot
race_ids = df_race_2018['raceId'].unique()
driver_ids = df_race_2018['driverId'].unique()
colors = sns.color_palette('hsv', len(race_ids))
color_map1 = dict(zip(race_ids, colors))
fig, axs = plt.subplots(1, 1, figsize=(30, 20))
sns.scatterplot(x='driverId_cat', y='milliseconds', data=driver_race_times, hue='raceId', palette=color_map1)

axs.set_xticks(range(len(driver_ids)))
axs.set_xticklabels(driver_ids)

plt.show()

#%%
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde

# Calculate the KDE
kde = gaussian_kde(driver_race_times['milliseconds'])
kde.set_bandwidth(bw_method=kde.factor / 2)
x = np.linspace(driver_race_times['milliseconds'].min(), driver_race_times['milliseconds'].max(), 1000)

# Find the local maxima
local_maxima = argrelextrema(kde(x), np.greater)

# Plot the KDE
plt.plot(x, kde(x), label='KDE')

# Mark the local maxima
for maxima_index in local_maxima[0]:
    maxima = x[maxima_index]
    plt.plot([maxima, maxima], [0, kde([maxima])[0]], 'r--')

plt.title('Kernel Density Estimate with Local Maxima')
plt.xlabel('Milliseconds')
plt.ylabel('Density')
plt.legend()
plt.show()

#%%
''' Text
Being that the data is non-globular and K-Means is not the best clustering algorithm for this data, we try with DBSCAN or Hierarchical Clustering.
Heirarchical clustering is a type of unsupervised machine learning algorithm used to cluster unlabeled data points.
'''
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

data_for_clustering = df_race_2018[['milliseconds', 'position']]

# Standardize the data
scaler = StandardScaler()
data_for_clustering['position'] = scaler.fit_transform(data_for_clustering[['position']])

#DBSCAN
DBSCAN_model = DBSCAN(eps=0.5, min_samples=5)
DBSCAN_model.fit(data_for_clustering)

labels =  DBSCAN_model.labels_

unique_labels = set(labels) - {-1}
colors = plt.cm.get_cmap('viridis', len(unique_labels))

for label in unique_labels:
    cluster_data = data_for_clustering[data_for_clustering['cluster'] == label]
    plt.scatter(cluster_data['milliseconds'], cluster_data['position'], color=colors(label), label=f'Cluster {label}')

outliers = data_for_clustering[data_for_clustering['cluster'] == -1]
plt.scatter(outliers['milliseconds'], outliers['position'], color='black', label='Outliers')

plt.xlabel('Milliseconds')
plt.ylabel('Position')

plt.show()

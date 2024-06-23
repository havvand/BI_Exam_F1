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
df_race_2021 = dataframes[2021]


#%% Histogram to show the distribution of the lap times
fig, axs = plt.subplots(1, 1, figsize=(30, 20))
sns.histplot(df_race_2021['milliseconds'], bins=150)
plt.xticks(np.arange(66000, 200000, 10000))

''' TEXT 
Based on the histograms, we can see the race data is not normally distributed across across the season.
The data seems to be multimodal, which suggests that there are two distinct subgroups within the data.
The indication of distinct subgroups could be due to various factors llike skill, track conditions or strategy..
'''

#%% Further analysis of the data to identify potential outliers.

print(df_race_2021.describe())

'''TEXT
The histogram and the describe functions both show, that there is an outlier at +500000 milliseconds and potientially at around 200000 milliseconds.
'''

df_race_2021 = df_race_2021[(df_race_2021['milliseconds'] <= 200000) & (df_race_2021['milliseconds'] >= 66000)]

fig, axs = plt.subplots(1, 1, figsize=(30, 20))
sns.histplot(df_race_2021['milliseconds'], bins=150)
plt.xticks(np.arange(66000, 200000, 1000))

df_race_2021.describe()

Q1 = df_race_2021['milliseconds'].quantile(0.25)
Q3 = df_race_2021['milliseconds'].quantile(0.75)
IQR = Q3 - Q1
print(Q1, Q3, IQR)
axs.fill_betweenx([0, 1000], Q1, Q3, color='red', alpha=0.3)



#%% Extracting the driver data
driver_race_times = df_race_2021.groupby('driverId')[['driverId', 'raceId', 'lap', 'milliseconds']].apply(lambda x: x)
driver_race_times['driverId_cat'] = pd.Categorical(df_race_2021['driverId']).codes

#%%
''' TEXT
 The threshold is set to 3 times the standard deviation for each race. This is a common method to identify outliers in a dataset.
'''
track_stats = df_race_2021.groupby('raceId')['milliseconds'].agg(['mean', 'std', 'min', 'max'])
threshold = 3
driver_race_times['outlier'] = driver_race_times.apply(lambda x: x['milliseconds'] > track_stats.loc[x['raceId']]['mean'] + threshold * track_stats.loc[x['raceId']]['std'], axis=1)


#%% Scatterplot
race_ids = df_race_2021['raceId'].unique()
driver_ids = df_race_2021['driverId'].unique()
colors = sns.color_palette('hsv', len(race_ids))
color_map1 = dict(zip(race_ids, colors))
fig, axs = plt.subplots(1, 1, figsize=(30, 20))

sns.scatterplot(x='driverId_cat', y='milliseconds', c=driver_race_times['outlier'], data=driver_race_times, hue='outlier')

axs.set_xticks(range(len(driver_ids)))
axs.set_xticklabels(driver_ids)

plt.show()

#%%
''' TEXT
Removing outliers from the data to get a better understanding of the distribution of the lap times.
'''
driver_race_times_no_outliers = driver_race_times[driver_race_times['outlier'] == False]
fig, axs = plt.subplots(1, 1, figsize=(30, 20))
sns.scatterplot(x='driverId_cat', y='milliseconds', data=driver_race_times_no_outliers, palette=color_map1)
axs.set_xticks(range(len(driver_ids)))
axs.set_xticklabels(driver_ids)
plt.show()

#%%
fig, axs = plt.subplots(1, 1, figsize=(30, 20))
sns.histplot(driver_race_times_no_outliers['milliseconds'], bins=150)
plt.xticks(np.arange(66000, 200000, 1000))

df_race_2021.describe()

Q1 = driver_race_times_no_outliers['milliseconds'].quantile(0.25)
Q3 = driver_race_times_no_outliers['milliseconds'].quantile(0.75)
IQR = Q3 - Q1
print(Q1, Q3, IQR)
axs.fill_betweenx([0, 1000], Q1, Q3, color='red', alpha=0.3)

#%%
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde

# Calculate the KDE
kde = gaussian_kde(driver_race_times['milliseconds'])
kde.set_bandwidth(bw_method=kde.factor / 2)
x = np.linspace(driver_race_times_no_outliers['milliseconds'].min(), driver_race_times_no_outliers['milliseconds'].max(), 1000)

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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from matplotlib.ticker import MaxNLocator
# Create a mapping from 'driverId_cat' to 'driverId'
driverId_mapping = dict(zip(driver_race_times_no_outliers['driverId_cat'], driver_race_times_no_outliers['driverId']))
# Create a new column 'driverId_cat' that represents the categorical version of 'driverId'
driver_race_times_no_outliers['driverId_cat'] = pd.Categorical(driver_race_times_no_outliers['driverId']).codes

kmeans = KMeans(n_clusters=16)
reshape = np.reshape(driver_race_times_no_outliers['milliseconds'].values, (-1, 1))
kmeans.fit(reshape)
driver_race_times_no_outliers['cluster'] = kmeans.predict(reshape)

silhouette_avg = silhouette_score(reshape,driver_race_times_no_outliers['cluster'])
print('Silhouette Score:', silhouette_avg)

chi_score = calinski_harabasz_score(reshape, driver_race_times_no_outliers['cluster'])
print('Calinski Harabasz Score:', chi_score)

db_score = davies_bouldin_score(reshape, driver_race_times_no_outliers['cluster'])
print('Davies Bouldin Score:', db_score)

# Get unique 'driverId_cat' values that have data
unique_driverId_cats = driver_race_times_no_outliers['driverId_cat'].unique()

# Set xticks to the unique 'driverId_cat' values that have data
plt.xticks(unique_driverId_cats)

# Set xtick labels to the corresponding 'driverId' values
plt.gca().set_xticklabels([driverId_mapping[i] for i in plt.gca().get_xticks()])

# Plot the clusters using 'driverId_cat' for the x-axis
plt.scatter(driver_race_times_no_outliers['raceId'], driver_race_times_no_outliers['milliseconds'], c=driver_race_times_no_outliers['cluster'], cmap='viridis')
plt.xlabel('Q1 Time')
plt.ylabel('Q2 Time')
plt.legend()
plt.title('K-Means Clustering of Q1 and Q2 times')

#%%
import plotly.graph_objects as go
import plotly.offline as pyo

# Create a trace for the scatter plot
trace = go.Scatter(
    x=driver_race_times_no_outliers['raceId'],
    y=driver_race_times_no_outliers['milliseconds'],
    mode='markers',
    marker=dict(
        size=10,
        color=driver_race_times_no_outliers['cluster'],
        colorscale='Viridis',
        showscale=True
    ),
    text=driver_race_times_no_outliers.apply(
        lambda row: f"Driver ID: {row['driverId']}<br>Race ID: {row['raceId']}<br>Lap Time: {row['milliseconds']}ms",
        axis=1
    ),  # Customize hover text
    hoverinfo='text'  # Show the custom text on hover
)

# Create the layout
layout = go.Layout(
    title='K-Means Clustering of Race Lap Times (2018)',
    xaxis=dict(
        title='Race ID',
        type='category',  # Treat Race ID as categorical
        tickmode='array',
        tickvals=driver_race_times_no_outliers['raceId'].unique(),  # Show all race IDs
        ticktext=driver_race_times_no_outliers['raceId'].unique()  # Display as text labels
    ),
    yaxis=dict(title='Lap Time (milliseconds)'),
)

# Create the figure and add the scatter plot
fig = go.Figure(data=[trace], layout=layout)

# Show the plot
# Show the plot
pyo.plot(fig)

#%%
''' Text 
Normalizing the lap times based on the fastest lap time. Easier to compare the driver performace.
This method  can help remove track specific biases and allow for a more accurate comparison of driver performance, but 
is sensitive to outliers.
'''
fastest_lap_per_race = driver_race_times_no_outliers.groupby('raceId')['milliseconds'].min().reset_index()
print(fastest_lap_per_race)
fastest_lap_per_race.columns = ['raceId', 'fastest_lap']

# Merge the fastest lap times with the original data
driver_race_times_no_outliers = pd.merge(driver_race_times_no_outliers, fastest_lap_per_race, on='raceId')

# Create and calculate the normalized lap time
driver_race_times_no_outliers['normalized_lap'] = driver_race_times_no_outliers['milliseconds'] / driver_race_times_no_outliers['fastest_lap']

#%%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from matplotlib.ticker import MaxNLocator
'''TEXT
The KMEANS method is not the best for this type of data, being it is not globular. But after trying DBSCAN (with no meaningful results) and Agglomerative Clustering, 
KMEANS was the best option
seemed to be the best option for this data (also based on my computer's processing power).
With scores of ;
Silhouette Score: 0.5257474847696829
Calinski Harabasz Score: 238033.90320716045
Davies Bouldin Score: 0.5245203232007064

It seems like a good clustering result, but it is important to remember that the data is not globular and the clusters are not too well-separated.
'''
# Create a mapping from 'driverId_cat' to 'driverId'
driverId_mapping = dict(zip(driver_race_times_no_outliers['driverId_cat'], driver_race_times_no_outliers['driverId']))
# Create a new column 'driverId_cat' that represents the categorical version of 'driverId'
driver_race_times_no_outliers['driverId_cat'] = pd.Categorical(driver_race_times_no_outliers['driverId']).codes

kmeans = KMeans(n_clusters=16)
reshape = np.reshape(driver_race_times_no_outliers['normalized_lap'].values, (-1, 1))
kmeans.fit(reshape)
driver_race_times_no_outliers['cluster'] = kmeans.predict(reshape)

silhouette_avg = silhouette_score(reshape,driver_race_times_no_outliers['cluster'])
print('Silhouette Score:', silhouette_avg)

chi_score = calinski_harabasz_score(reshape, driver_race_times_no_outliers['cluster'])
print('Calinski Harabasz Score:', chi_score)

db_score = davies_bouldin_score(reshape, driver_race_times_no_outliers['cluster'])
print('Davies Bouldin Score:', db_score)
#Get unique 'driverId_cat' values that have data
unique_driverId_cats = driver_race_times_no_outliers['driverId_cat'].unique()
#Plot the clusters using 'driverId_cat' for the x-axis
plt.scatter(driver_race_times_no_outliers['raceId'], driver_race_times_no_outliers['milliseconds'], c=driver_race_times_no_outliers['cluster'], cmap='viridis')
plt.xlabel('Race Id')
plt.ylabel('Lap Time')
plt.legend()
plt.title('K-Means Clustering of Q1 and Q2 times')
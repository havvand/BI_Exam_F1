import fastf1 as ff1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Qualifying and lap time data will be analysed in the following. Starting with loading, cleaning and then exploration.

quali_data = pd.read_csv('data/qualifying.csv')

# Cleaning the qualifying data.

print('QUALIFICATION DATA INFO:') # Check data types and for missing values
print(quali_data.info())
print(quali_data.notnull().sum())

quali_data_2018_2023 = quali_data[(quali_data['raceId'] >= 989) & (quali_data['raceId'] <= 1110)]

quali_data_2018_2023.rename(columns={'position': 'quali_position'}, inplace=True)
quali_data_2018_2023.loc[:, 'q1'] = quali_data_2018_2023['q1'].replace('\\N', np.nan)
quali_data_2018_2023.loc[:, 'q2'] = quali_data_2018_2023['q2'].replace('\\N', np.nan)
quali_data_2018_2023.loc[:, 'q3'] = quali_data_2018_2023['q3'].replace('\\N', np.nan)

quali_data_2018_2023.notnull().sum()

# Converting all time-data to milliseconds

def convert_time_to_ms(time_str):
    if pd.isna(time_str):
        return np.nan
    if ':' not in time_str:
        return None
    mins, time = time_str.split(':')
    if '.' not in time:
        secs = time
        ms = 0
    secs, ms = time.split('.')
    return (float(mins) * 60 + float(secs)) * 1000 + float(ms)


for column in ['q1', 'q2', 'q3']:
    quali_data_2018_2023.loc[:, column] = quali_data_2018_2023[column].apply(convert_time_to_ms)

#%% Seperating the data into Seasons 2018-2023

''' TEXT 
We decided to divide the data into seasons, to get a better understanding of the performance of the drivers across the seasons.
This allows for a more detailed analysis of the data and the performance of the drivers. Because circumstances can change from season to season.
Improvements and changes in the cars, drivers and teams can be better understood by looking at the data in this way.
Identifying outliers and applying statistical methods will be more robust when done by seasons
'''

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
unique_raceIds = quali_data_2018_2023['raceId'].unique()
# Looping through the dictionary to extract the data for each season
for year, (start, end) in year_intervals.items():
    dataframes[year] = quali_data_2018_2023[(quali_data_2018_2023['raceId'] >= start) & (quali_data_2018_2023['raceId'] <= end)]


#%% Boxplot of Q1, Q2 and Q3 times

df_2018_quali = dataframes[2018]

#%% Exploration and anaylsis of the quali_data_2018-2023 - Histogram of Q1, Q2 and Q3 times to explore the distribution of the data
print(quali_data_2018_2023.describe())

# Histogram of Q1, Q2 and Q3 times to explore the distribution of the data
fig, axs = plt.subplots(1, 3, figsize=(30, 20))
sns.histplot(df_2018_quali['q1'], bins=47, ax=axs[0])
sns.histplot(df_2018_quali['q2'], bins=47, ax=axs[1])
sns.histplot(df_2018_quali['q3'], bins=47, ax=axs[2])

''' TEXT 
Based on the histograms, we can see the data is not normally distributed across Q1, Q2 and Q3.
The data is multi-modal, which suggests that there are distinct subgroups within the data.
The indication of distinct subgroups could be due to various factors llike skill, track conditions or strategy..
Some outliers are also clearly identified in the data, which can be removed to get a more accurate picture of the data.
'''

#%% Further investigation of histogram. Removing outliers and calculating the IQR for Q1, Q2 and Q3 times

# Removing outliers, based on the histogram above 120,000 ms and below 54,000 ms. This gives a more accurate picture of the performances and a more accurate Standard Deviation

df_2018_quali = df_2018_quali[((df_2018_quali['q1'].isna()) | ((df_2018_quali['q1'] <= 106000) & (df_2018_quali['q1'] >= 53000))) &
                              ((df_2018_quali['q2'].isna()) | ((df_2018_quali['q2'] <= 106000) & (df_2018_quali['q2'] >= 53000))) &
                              ((df_2018_quali['q3'].isna()) | ((df_2018_quali['q3'] <= 106000) & (df_2018_quali['q3'] >= 53000)))]
df_2018_quali = df_2018_quali[(df_2018_quali['driverId'] != 850) & (df_2018_quali['driverId'] != 851)]

fig, axs = plt.subplots(1, 3, figsize=(30, 20))
sns.histplot(df_2018_quali['q1'], bins=21, ax=axs[0])
sns.histplot(df_2018_quali['q2'], bins=21, ax=axs[1])
sns.histplot(df_2018_quali['q3'], bins=21, ax=axs[2])

# Calculating the IQR for Q1, Q2 and Q3 times
Q1R1_q1 = df_2018_quali['q1'].quantile(0.25)
Q1R1_q3 = df_2018_quali['q1'].quantile(0.75)
iqr_1 = Q1R1_q3 - Q1R1_q1

Q1R2_q1 = df_2018_quali['q2'].quantile(0.25)
Q1R2_q3 = df_2018_quali['q2'].quantile(0.75)
iqr_2 = Q1R2_q3 - Q1R2_q1

Q1R3_q1 = df_2018_quali['q3'].quantile(0.25)
Q1R3_q3 = df_2018_quali['q3'].quantile(0.75)
iqr_3 = Q1R2_q3 - Q1R2_q1

print('IQR_1', iqr_1)
print('IQR_2', iqr_2)
print('IQR_3', iqr_3)
print(Q1R1_q1, ' - ', Q1R1_q3)
print(Q1R2_q1, ' - ', Q1R2_q3)
print(Q1R3_q1, ' - ', Q1R3_q3)


# Adding the IQR to the histograms
axs[0].fill_betweenx([0, plt.gca().get_ylim()[1]], Q1R1_q1, Q1R1_q3, color='red', alpha=0.2)
axs[1].fill_betweenx([0, plt.gca().get_ylim()[1]], Q1R2_q1, Q1R2_q3, color='red', alpha=0.2)
axs[2].fill_betweenx([0, plt.gca().get_ylim()[1]], Q1R3_q1, Q1R3_q3, color='red', alpha=0.2)

plt.title('Histogram of Q1, Q2 and Q3 times')
plt.ylabel('Frequency')

''' TEXT 
The IQR for Q1, Q2 and Q3 are 18,3ms 19.7ms and 19.7ms respectively.

The Historgram for Q1 shows that the inter quartile range (IQR) which represents 50% of Q1 lap times, falls between 75,835 and 94,140 milliseconds.
This suggests, that a significant portion of drivers across the grid exhibit similar performance within this range, with half their laps in this range.

18+ seconds are quite significant in Formula One, where the difference between pole position and 10th place can be less than a second. Due to the fact
that we are looking at the whole season, we cant include weather conditions, track conditions, car performance etc., which can explain the high IQR.

The STD is 11.37 seconds, which is also quite high. This shows us, that on average the individual lap times deviate from the mean by 11.37 seconds.
Again a significant amount of time in Formula One. The high STD can also be explained by the fact we are looking at the whole season.

Taking into consideration we are looking at a whole season, we can still gain insights into the performance of the drivers and the grid as a whole.

E.g. Q1 Shows the widest range of laptimes, with some slow outliers. This can be due to warm-up laps, traffic or other factors.
Q2 and Q3 show a more narrow range of lap times, which is expected, as the drivers are pushing for the best possible lap times. 
Q2 and Q3 are more competitive, with the fastest drivers in the grid competing for the top positions.

To gain insights into the subgroups, we can use clustering techniques to group the drivers based on their Q1, Q2 and Q3 times.
'''
plt.show()

#%% Collecting all the data for driver qualifying times in 2018
driver_quali_times = df_2018_quali.groupby('driverId')[['driverId', 'raceId', 'q1', 'q2', 'q3']].apply(lambda x: x)
driver_quali_times['driverId_cat'] = pd.Categorical(driver_quali_times['driverId']).codes

#%% Calulating the mean and standard deviation for Q1, Q2 and Q3 times and plotting the data in a scatterplot
''' TEXT 
The scatterplot shows the distribution of the data and the mean for each Q1, Q2 and Q3 times.
This visualises each drivers performance across the season and allows for a comparison of the drivers.
Note: There i no mode in the data, being that no lap time is repeated. - The data is amodal. '''

race_989 = df_2018_quali[df_2018_quali['raceId'] == 989]
race_990 = df_2018_quali[df_2018_quali['raceId'] == 990]
mean_990 = race_990['q1'].mean()
mean_989 = race_989['q1'].mean()
print('989', mean_989)


mean_q1 = df_2018_quali['q1'].mean()
print('Mean Q1:', mean_q1)
median_q1 = df_2018_quali['q1'].median()
print('Median Q1:', median_q1)
std_deviation_q1 = df_2018_quali['q1'].std()
print('Std Deviation Q1:', std_deviation_q1)
standard_error_q1 = df_2018_quali['q1'].sem()
print('Standard Error:', standard_error_q1)


mean_q2 = df_2018_quali['q2'].mean()
print('Mean Q2:', mean_q2)
median_q2 = df_2018_quali['q2'].median()
print('Median Q2:', median_q2)
std_deviation_q2 = df_2018_quali['q2'].std()
print('Std Deviation Q2:', std_deviation_q2)
standard_error_q2 = df_2018_quali['q2'].sem()
print('Standard Error:', standard_error_q2)


mean_q3 = df_2018_quali['q3'].mean()
print('Mean Q3:', mean_q3)
median_q3 = df_2018_quali['q3'].median()
print('Median Q3:', median_q3)
std_deviation_q3 = df_2018_quali['q3'].std()
print('Std Deviation Q3:', std_deviation_q3)
standard_error_q3 = df_2018_quali['q3'].sem()
print('Standard Error:', standard_error_q3)

race_ids = df_2018_quali['raceId'].unique()
driver_ids = df_2018_quali['driverId'].unique()
colors = sns.color_palette('hsv', len(race_ids))
color_map1 = dict(zip(race_ids, colors))
fig, axs = plt.subplots(1, 3, figsize=(10, 10))

sns.scatterplot(x='driverId_cat', y='q1', data=driver_quali_times, ax=axs[0], hue='raceId', palette=color_map1)
sns.scatterplot(x='driverId_cat', y='q2', data=driver_quali_times, ax=axs[1], hue='raceId', palette=color_map1)
sns.scatterplot(x='driverId_cat', y='q3', data=driver_quali_times, ax=axs[2], hue='raceId', palette=color_map1)

axs[0].axhline(y=mean_q1, color='r', linestyle='--')
axs[0].axhline(y=mean_989, color='r', linestyle='-.')
axs[0].axhline(y=mean_990, color='orange', linestyle='-.')
axs[1].axhline(y=mean_q2, color='r', linestyle='--')
axs[2].axhline(y=mean_q3, color='r', linestyle='--')

for ax in axs:
    ax.set_xticks(range(len(driver_ids)))
    ax.set_xticklabels(driver_ids)
#plt.xticks(range(len(driver_ids)), driver_ids)
plt.show()

#%%
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde

df_2018_quali = df_2018_quali.dropna(subset=['q1'])
#%%
df_2018_quali['q1'] = pd.to_numeric(df_2018_quali['q1'], errors='coerce')


# Calculate the KDE
kde = gaussian_kde(df_2018_quali['q1'])
kde.set_bandwidth(bw_method=kde.factor / 2)
x = np.linspace(df_2018_quali['q1'].min(), df_2018_quali['q1'].max(), 1000)

# Find the local maxima
local_maxima = argrelextrema(kde(x), np.greater)

# Plot the KDE
plt.plot(x, kde(x), label='KDE')

# Mark the local maxima
for maxima_index in local_maxima[0]:
    maxima = x[maxima_index]
    plt.plot([maxima, maxima], [0, kde([maxima])[0]], 'r--')

plt.title('Kernel Density Estimate with Local Maxima')
plt.xlabel('Q1 Time')
plt.ylabel('Density')
plt.legend()
plt.show()

#%% Boxplot of Q1, Q2 and Q3 times
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=10)
reshape = np.reshape(df_2018_quali['q1'].values, (-1, 1))
gmm.fit(reshape)

means = gmm.means_
variances = gmm.covariances_
weights = gmm.weights_

print('Means:', means)
print('Variances:', variances)
print('Weights:', weights)

# Testing for the optimal number of components - According to BIC and AIC the optimal number of components is 10.
# The BIC and AIC values are the lowest for 10 components, but knowing the data, we can see that 10 components is too few.
'''for i in range(1, gmm.n_components+1):
    gmm = GaussianMixture(n_components=i)
    reshape = np.reshape(df_2018_quali['q1'].values, (-1, 1))
    gmm.fit(reshape)
    bic =  gmm.bic(reshape)
    print('BIC for: ', i, ' components ',  bic)
    aic = gmm.aic(reshape)
    print('AIC for : ',i , ' components ' , aic)'''


#%% Plotting the KDE plot or histogram with the GMM components
''' TEXT 
The GMM components are plotted on top of the KDE plot or histogram to show the distribution of the data and the components.
This tells us, that the data is not normally distributed and that there are distinct subgroups in the data.
Based on a visual inspection, there are around 5 subgroups in the data, which can be further analysed using clustering techniques.
Where the curve is low, we see less probability of the data being in that range.
As expected the curve is low around the fastest and slowest speeds. This is because there are fewer drivers in these ranges.
'''

from scipy.stats import norm
# Generate the KDE plot or histogram. Bins set by Squareroot of the number of data points
sns.histplot(df_2018_quali['q1'], bins=20, kde=True)

# For each component in the GMM
print(gmm.n_components)
for i in range(gmm.n_components):
    # Generate a Gaussian distribution using the mean and variance of the component
    gauss = norm(loc=means[i][0], scale=np.sqrt(variances[i][0][0]))

    # Generate x values
    x = np.linspace(df_2018_quali['q1'].min(), df_2018_quali['q1'].max(), 1000)

    # Plot the Gaussian distribution
    plt.plot(x, weights[i] * gauss.pdf(x), 'r')

plt.title('Histogram of Q1 times with GMM components')
plt.xlabel('Time in milliseconds')
plt.ylabel('Frequency')
plt.show()

#%% K-Means Clustering
''' TEXT
Inspecting the clusters - starting with 10 clusters - of the data using K-Means clustering, we can quickly identify a patteren. 
The data seems to fit with the different tracks which is used in the season. Dividing them into 20 clusters, seems to fit well with the data. 

The Silhouette Score and Calinski Harabasz Score are used to evaluate the clustering. The Silhouette Score is above the threshold of 0.5, which is good.
The Calinski Harabasz Score is also above the threshold of 1000, which is good. This indicates that the clusters are well separated and that the data fits well with the clusters.
The Davies Bouldin Score is also used to evaluate the clustering. The score is below 0.5, which is good. This indicates that the clusters are well separated and that the data fits well with the clusters.

The metrics seems to indicate that the clustering is good and that the data fits well with the clusters and each cluster cooresponds to a different track.
'''

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

df_2018_quali = df_2018_quali.dropna(subset=['q1'])
kmeans = KMeans(n_clusters=20)
reshape = np.reshape(df_2018_quali['q1'].values, (-1, 1))
kmeans.fit(reshape)
df_2018_quali['cluster'] = kmeans.predict(reshape)

silhouette_avg = silhouette_score(reshape,  df_2018_quali['cluster'])
print('Silhouette Score:', silhouette_avg)

chi_score = calinski_harabasz_score(reshape, df_2018_quali['cluster'])
print('Calinski Harabasz Score:', chi_score)

db_score = davies_bouldin_score(reshape, df_2018_quali['cluster'])
print('Davies Bouldin Score:', db_score)

# Plot the clusters
plt.scatter(df_2018_quali['q1'], df_2018_quali['q2'], c=df_2018_quali['cluster'], cmap='viridis')
plt.xlabel('Q1 Time')
plt.ylabel('Q2 Time')
plt.title('K-Means Clustering of Q1 and Q2 times')

#%%
import plotly.graph_objects as go
import plotly.offline as pyo

# Create a trace for the scatter plot
trace = go.Scatter(
    x = df_2018_quali['q1'],
    y = df_2018_quali['q2'],
    mode = 'markers',
    marker = dict(
        size = 10,
        color = df_2018_quali['cluster'], # set color equal to a variable
        colorscale = 'Viridis', # one of plotly colorscales
        showscale = True
    ),
    text = df_2018_quali.raceId.astype(str) + '<br>' + 'DriverID: '+  df_2018_quali.driverId.astype(str) # This will be displayed when you hover over a point
)

# Create a trace for the scatter plot for Q3
trace2 = go.Scatter(
    x = df_2018_quali['q1'],
    y = df_2018_quali['q3'],
    mode = 'markers',
    marker = dict(
        size = 10,
        color = df_2018_quali['cluster'], # set color equal to a variable
        colorscale = 'Viridis', # one of plotly colorscales
        showscale = True
    ),
    text = df_2018_quali.raceId.astype(str) + '<br>' + 'DriverID: ' + df_2018_quali.driverId.astype(str) + '<br>' + 'Q3', # This will be displayed when you hover over a point
    name = 'Q1 vs Q3'
)

data = [trace, trace2]

# Create the layout
layout = go.Layout(
    title = 'K-Means Clustering of Q1 and Q2 times',
    xaxis = dict(title = 'Q1 Time'),
    yaxis = dict(title = 'Q2 and Q3 Time'),
    hovermode = 'closest'
)

# Create the figure and add the scatter plot
fig = go.Figure(data=data, layout=layout)

# Show the plot
pyo.plot(fig)

#%%
test_heatmap = test.loc[test['raceId'] == 1106]
print(test_heatmap.dtypes)

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(test_heatmap.corr(), annot=False, ax=ax)
corr = test_heatmap.corr()
## Add the annotation manually
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        text = ax.text(j + 0.5, i + 0.5, round(corr.iloc[i, j], 2),
                       ha="center", va="center", color="w")
plt.show()

#%%
plt.scatter(test_heatmap['car_number'], test_heatmap['q3'], label='Q3-time')
plt.xlabel('Car Number')
plt.ylabel('Q Time')
plt.title('Scatter plot of Car Number vs Q Time')

plt.scatter(test_heatmap['car_number'], test_heatmap['q2'], label='Q2-time')
plt.xlabel('Car Number')
plt.ylabel('Q Time')
plt.title('Scatter plot of Car Number vs Q Time')

plt.scatter(test_heatmap['car_number'], test_heatmap['q1'], label='Q1-time')
plt.xlabel('Car Number')
plt.ylabel('Q Time')
plt.title('Scatter plot of Car Number vs Q Time')

plt.scatter(test_heatmap['car_number'], test_heatmap['fastestLapTime'], label='Fastest Lap Time')

plt.legend()
plt.show()


#%%
w_data = session = ff1.get_session(2023, 'Canada', 'Q')
session.load(weather=True)

#%%
start_time = session.date
fate = session.session_start_time
print(start_time, '-' ,fate)
print(session.session_info)
session.weather_data.head()

#%%
session.weather_data.to_csv('data/weather_data_Canada_2021_1106.csv', index=False)
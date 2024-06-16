import fastf1.plotting
import matplotlib.pyplot as plt
import pandas
import pandas as pd
import seaborn as sns
import numpy as np

import fastf1 as ff1
import fastf1.plotting as ff1plt

# for drawing dendogram
import scipy.cluster.hierarchy as ch
from scipy.spatial.distance import cdist

# for creating a model
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

circuit_data = pd.read_csv('data/circuits.csv')
race_data = pd.read_csv('data/races.csv')

# Select data from 2018 to 2023 - "Modern Era"
races_2018_2023 = race_data[(race_data['year'] >= 2018) & (race_data['year'] <= 2023)]
races_2018_2023.drop(columns=['fp1_date', 'fp1_time', 'fp2_date', 'fp2_time', 'fp3_date', 'fp3_time', 'quali_date', 'quali_time', 'sprint_date', 'sprint_time', 'url'], inplace=True)
# print(races_2018_2023.head())

# Extracting the unique circuit IDs for 2018-2023 and enriching them, so we can use them to get other relevant data
unique_circuit_ids = pd.DataFrame(races_2018_2023['circuitId'].drop_duplicates())
race_circuit_data = unique_circuit_ids.merge(circuit_data, on="circuitId")

print(race_circuit_data.head())



#%% Combining qualifying and raceing data into a single dataframe for 2018-2023.
quali_data = pd.read_csv('data/qualifying.csv')
quali_data.rename(columns={'position': 'quali_position'}, inplace=True)
quali_data['q2'] = quali_data['q2'].replace('\\N', np.nan)
quali_data['q3'] = quali_data['q3'].replace('\\N', np.nan)

results_data = pd.read_csv('data/results.csv')
results_data.drop(['time'], inplace=True, axis=1)
results_data['positionText'] = results_data['positionText'].replace({'R': 0, 'D': 00})
results_data[['raceId', 'driverId', 'constructorId', 'grid', 'position', 'positionOrder', 'points', 'laps', 'milliseconds',
              'fastestLap', 'rank', 'fastestLapTime', 'fastestLapSpeed']] = results_data[['raceId', 'driverId', 'constructorId', 'grid', 'position', 'positionOrder', 'points', 'laps', 'milliseconds', 'fastestLap', 'rank', 'fastestLapTime', 'fastestLapSpeed']].replace('\\N', np.nan)

test = pd.merge(quali_data, results_data, on=['raceId', 'driverId', 'constructorId'])
test.drop(['number_x'], inplace=True, axis=1)
test.rename(columns={'number_y': 'car_number'}, inplace=True)

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


for column in ['q1', 'q2', 'q3','fastestLapTime']:
    test[column] = test[column].apply(convert_time_to_ms)


#%%
sns.boxplot(x='raceId', y='q1', data=quali_data)
plt.show()

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

import fastf1.plotting
import matplotlib.pyplot as plt
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
race_data_2018_2023 = race_data[(race_data['year'] >= 2018) & (race_data['year'] <= 2023)]
race_data_2018_2023.drop(columns=['fp1_date', 'fp1_time', 'fp2_date', 'fp2_time', 'fp3_date', 'fp3_time', 'quali_date', 'quali_time', 'sprint_date', 'sprint_time'], inplace=True)
print(race_data_2018_2023.head())

#%% Extracting the unique circuit IDs for 2018-2023 and enriching them, so we can use them to get other relevant data
unique_circuit_ids = pd.DataFrame(race_data_2018_2023['circuitId'].drop_duplicates())
unique_circuit_ids = unique_circuit_ids.merge(circuit_data, on="circuitId")
print(unique_circuit_ids)

#%%

#%%
quali_data = pd.read_csv('data/qualifying.csv')
quali_data['q2'] = quali_data['q2'].replace('\\N', np.nan)
quali_data['q3'] = quali_data['q2'].replace('\\N', np.nan)
print(quali_data.head())

sns.scatterplot(x='driverId', y='position', data=quali_data)

#%%
session = ff1.get_session(2021, 'Bahrain', 'R')
session.load(weather=True)
#%%
session.weather_data.to_csv('data/weather_data_Bahrain_2021.csv', index=False)

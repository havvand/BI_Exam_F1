import fastf1 as ff1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# for drawing dendogram

# for creating a model

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




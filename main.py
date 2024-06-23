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

#%% Seperating the data into Seasons 2018-2023

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

    # Race data
    race_data = pd.read_csv('data/lap_times.csv')
    print(race_data.head())

    ## Separate the data from 2018 to 2023 into a new dataframe
    race_data_2018_2023 = race_data[(race_data['raceId'] >= 989) & (race_data['raceId'] <= 1110)]

    '''RACE DATA INFO'''
    print(race_data_2018_2023.info())
    print(race_data_2018_2023.notnull().sum())

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
dataframes_quali = {}
# Extracting each unique raceId for the seasons
unique_raceIds = quali_data_2018_2023['raceId'].unique()
# Looping through the dictionary to extract the data for each season
for year, (start, end) in year_intervals.items():
    dataframes_quali[year] = quali_data_2018_2023[(quali_data_2018_2023['raceId'] >= start) & (quali_data_2018_2023['raceId'] <= end)]
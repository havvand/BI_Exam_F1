import _2018_quali
import _2019_quali
import _2020_quali
import _2021_quali
import _2022_quali
import _2018_race
import _2019_race
import _2020_race
import _2021_race
import _2022_race


df_2018_quali = _2018_quali
df_2019_quali = _2019_quali
df_2020_quali = _2020_quali
df_2021_quali = _2021_quali
df_2022_quali = _2022_quali

df_2018_race = _2018_race
df_2019_race = _2019_race
df_2020_race = _2020_race
df_2021_race = _2021_race
df_2022_race = _2022_race
#%%
quali_2018 = df_2018_quali.df_2018_quali
quali_2019 = df_2019_quali.df_2019_quali
quali_2020 = df_2020_quali.df_2020_quali
quali_2021 = df_2021_quali.df_2021_quali
quali_2022 = df_2022_quali.df_2022_quali

race_2018 = df_2018_race.driver_race_times_no_outliers
race_2019 = df_2019_race.driver_race_times_no_outliers
race_2020 = df_2020_race.driver_race_times_no_outliers
race_2021 = df_2021_race.driver_race_times_no_outliers
race_2022 = df_2022_race.driver_race_times_no_outliers

#%%
'''TEXT
Inner Join: 
-  We use an inner join (how='inner') to ensure that only the drivers who participated in both qualifying and the race are included in the merged dataset.'''
import pandas as pd
merged_data = {}
for year in [2018, 2019, 2020, 2021, 2022]:
    quali_df = globals()[f'quali_{year}']
    race_df = globals()[f'race_{year}']

    quali_df = quali_df.rename(columns={'cluster': 'quali_cluster'})
    race_df = race_df.rename(columns={'cluster': 'race_cluster'})

    merged_data[year] = pd.merge(quali_df, race_df, on=['raceId', 'driverId'], how='inner')

all_merged_data = pd.concat(merged_data.values(), ignore_index=True)

print(all_merged_data.head())

#%%
import seaborn as sns
import numpy as np
all_merged_data = all_merged_data.replace(pd.NA, np.nan)
sns.heatmap(all_merged_data.corr(), annot=True, fmt='.2f', cmap='coolwarm')

#%%
sns.scatterplot(data=all_merged_data, x='race_cluster', y='quali_cluster')

#%%
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


all_merged_data2 = pd.concat(merged_data.values(), ignore_index=True)
all_merged_data['position'] = all_merged_data['position'].astype(int)

# 2. Feature Engineering
features = ['quali_position', 'race_cluster', 'quali_cluster', 'constructorId', 'driverId']  # Example

#   b. Encode Categorical Variables: If you have categorical features
all_merged_data2 = pd.get_dummies(all_merged_data2, columns=['constructorId'])

#   c. Handle Missing Values: Fill in missing values with appropriate strategies
all_merged_data2 = all_merged_data2.fillna(0)  # Example: Fill with 0

# 3. Define Target Variable:
target_variable = 'position'  # Example

# 4. Split Data into Training and Testing Sets:
X = all_merged_data2[features]
y = all_merged_data[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Random Forest Model:
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 6. Make Predictions on Test Set:
y_pred = rf_model.predict(X_test)

# 7. Evaluate Model Performance:
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

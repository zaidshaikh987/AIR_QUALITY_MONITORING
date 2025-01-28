# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load datasets
df_pollutants = pd.read_csv('C:/Users/MD.ZAID SHAIKH/Desktop/AIR_QUALITY_MONITORING/Backend/data/factors_Nashik.csv')
df_weather = pd.read_csv('C:/Users/MD.ZAID SHAIKH/Desktop/AIR_QUALITY_MONITORING/Backend/data/pollutants_Nashik.csv')

# Sort both datasets by date in ascending order
df_pollutants = df_pollutants.sort_values('date')
df_weather = df_weather.sort_values('date')

# Convert date columns to datetime format
df_pollutants['date'] = pd.to_datetime(df_pollutants['date'])
df_weather['date'] = pd.to_datetime(df_weather['date'])

# Find common dates
common_dates = set(df_pollutants['date']).intersection(set(df_weather['date']))

# Filter both datasets to keep only common dates
df_pollutants = df_pollutants[df_pollutants['date'].isin(common_dates)]
df_weather = df_weather[df_weather['date'].isin(common_dates)]

# Merge datasets on date
df_combined = pd.merge(df_pollutants, df_weather, on='date', how='inner')

# Display the first few rows of the combined dataset
df_combined.head()

# Print the number of rows in each dataset
print("Pollutant Data Rows:", len(df_pollutants))
print("Weather Data Rows:", len(df_weather))
print("Combined Data Rows:", len(df_combined))

# Check data types and null values
print(df_combined.info())

# Convert pollutant columns to numeric
pollutant_columns = [' pm25', ' pm10', ' o3', ' no2', ' so2', ' co']
for col in pollutant_columns:
    df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')

# Check for missing values
print(df_combined[pollutant_columns].isnull().sum())

# Fill missing values using interpolation
df_combined[pollutant_columns] = df_combined[pollutant_columns].interpolate(method='linear', limit_direction='both')

# Double-check for any remaining NaNs
print(df_combined[pollutant_columns].isnull().sum())

# Verify data types
print(df_combined.dtypes)

# Temporal features
df_combined['day_of_week'] = df_combined['date'].dt.dayofweek
df_combined['month'] = df_combined['date'].dt.month
df_combined['season'] = (df_combined['month'] % 12 + 3) // 3  # 1: Winter, 2: Spring, etc.

# Lag features for pollutants
df_combined['pm25_lag1'] = df_combined[' pm25'].shift(1)
df_combined['pm25_lag2'] = df_combined[' pm25'].shift(2)

# 7-day rolling average for PM2.5
df_combined['pm25_7day_avg'] = df_combined[' pm25'].rolling(window=7).mean()

# Drop rows with NaN values
df_combined.dropna(inplace=True)

# Add a time index
df_combined['time_idx'] = (df_combined['date'] - df_combined['date'].min()).dt.days

# Display the first few rows of the updated dataset
print(df_combined.head())

# Check data types and null values again
print(df_combined.info())

# Define breakpoints for AQI calculation
breakpoints = {
    "pm25": [
        {"low": 0, "high": 12, "aqi_low": 0, "aqi_high": 50},
        {"low": 12.1, "high": 35.4, "aqi_low": 51, "aqi_high": 100},
        {"low": 35.5, "high": 55.4, "aqi_low": 101, "aqi_high": 150},
        {"low": 55.5, "high": 150.4, "aqi_low": 151, "aqi_high": 200},
        {"low": 150.5, "high": 250.4, "aqi_low": 201, "aqi_high": 300},
        {"low": 250.5, "high": 350.4, "aqi_low": 301, "aqi_high": 400},
        {"low": 350.5, "high": 500.4, "aqi_low": 401, "aqi_high": 500}
    ],
    "pm10": [
        {"low": 0, "high": 54, "aqi_low": 0, "aqi_high": 50},
        {"low": 55, "high": 154, "aqi_low": 51, "aqi_high": 100},
        {"low": 155, "high": 254, "aqi_low": 101, "aqi_high": 150},
        {"low": 255, "high": 354, "aqi_low": 151, "aqi_high": 200},
        {"low": 355, "high": 424, "aqi_low": 201, "aqi_high": 300},
        {"low": 425, "high": 504, "aqi_low": 301, "aqi_high": 400},
        {"low": 505, "high": 604, "aqi_low": 401, "aqi_high": 500}
    ],
    "o3": [
        {"low": 0, "high": 54, "aqi_low": 0, "aqi_high": 50},
        {"low": 55, "high": 70, "aqi_low": 51, "aqi_high": 100},
        {"low": 71, "high": 85, "aqi_low": 101, "aqi_high": 150},
        {"low": 86, "high": 105, "aqi_low": 151, "aqi_high": 200},
        {"low": 106, "high": 200, "aqi_low": 201, "aqi_high": 300},
        {"low": 201, "high": 300, "aqi_low": 301, "aqi_high": 400},
        {"low": 301, "high": 400, "aqi_low": 401, "aqi_high": 500}
    ],
    "no2": [
        {"low": 0, "high": 53, "aqi_low": 0, "aqi_high": 50},
        {"low": 54, "high": 100, "aqi_low": 51, "aqi_high": 100},
        {"low": 101, "high": 360, "aqi_low": 101, "aqi_high": 150},
        {"low": 361, "high": 649, "aqi_low": 151, "aqi_high": 200},
        {"low": 650, "high": 1249, "aqi_low": 201, "aqi_high": 300},
        {"low": 1250, "high": 1649, "aqi_low": 301, "aqi_high": 400},
        {"low": 1650, "high": 2049, "aqi_low": 401, "aqi_high": 500}
    ],
    "so2": [
        {"low": 0, "high": 35, "aqi_low": 0, "aqi_high": 50},
        {"low": 36, "high": 75, "aqi_low": 51, "aqi_high": 100},
        {"low": 76, "high": 185, "aqi_low": 101, "aqi_high": 150},
        {"low": 186, "high": 304, "aqi_low": 151, "aqi_high": 200},
        {"low": 305, "high": 604, "aqi_low": 201, "aqi_high": 300},
        {"low": 605, "high": 804, "aqi_low": 301, "aqi_high": 400},
        {"low": 805, "high": 1004, "aqi_low": 401, "aqi_high": 500}
    ],
    "co": [
        {"low": 0, "high": 4.4, "aqi_low": 0, "aqi_high": 50},
        {"low": 4.5, "high": 9.4, "aqi_low": 51, "aqi_high": 100},
        {"low": 9.5, "high": 12.4, "aqi_low": 101, "aqi_high": 150},
        {"low": 12.5, "high": 15.4, "aqi_low": 151, "aqi_high": 200},
        {"low": 15.5, "high": 30.4, "aqi_low": 201, "aqi_high": 300},
        {"low": 30.5, "high": 40.4, "aqi_low": 301, "aqi_high": 400},
        {"low": 40.5, "high": 50.4, "aqi_low": 401, "aqi_high": 500}
    ]
}

# Function to calculate AQI for a pollutant based on its concentration
def calculate_aqi_for_pollutant(concentration, pollutant):
    if pd.isna(concentration):  # Handle missing values
        return None
    for bp in breakpoints[pollutant]:
        if bp["low"] <= concentration <= bp["high"]:
            # Linear interpolation to calculate AQI
            return ((bp["aqi_high"] - bp["aqi_low"]) / (bp["high"] - bp["low"])) * (concentration - bp["low"]) + bp["aqi_low"]
    return None  # Return None if concentration is outside all breakpoints  

# Strip leading/trailing spaces from column names
df_combined.columns = df_combined.columns.str.strip()

# Debug: Check if column names match
print("Column names in df_combined:", df_combined.columns)

# Create a new DataFrame to store AQI calculations
df_with_aqi = df_combined.copy()

# Example pollutant column names in df_combined
pollutants = ["pm25", "pm10", "o3", "no2", "so2", "co"]  # Update if column names are different

# Debug: Check if pollutant columns exist
missing_columns = [col for col in pollutants if col not in df_combined.columns]
if missing_columns:
    print(f"Missing pollutant columns in dataset: {missing_columns}")
else:
    print("All pollutant columns are present.")

# Ensure all pollutant columns are numeric
for pollutant in pollutants:
    if pollutant in df_combined.columns:
        df_with_aqi[pollutant] = pd.to_numeric(df_with_aqi[pollutant], errors='coerce')

# Debug: Check for NaN values
print("Number of NaN values per column:")
print(df_with_aqi[pollutants].isna().sum())

# Calculate AQI for each pollutant across all rows in df_with_aqi
for pollutant in pollutants:
    if pollutant in df_with_aqi.columns:
        df_with_aqi[f"{pollutant}_aqi"] = df_with_aqi[pollutant].apply(
            lambda x: calculate_aqi_for_pollutant(x, pollutant)
        )

# Calculate overall AQI as the maximum of all individual AQI values for each row
aqi_columns = [f"{pollutant}_aqi" for pollutant in pollutants if f"{pollutant}_aqi" in df_with_aqi.columns]
if aqi_columns:
    df_with_aqi["overall_aqi"] = df_with_aqi[aqi_columns].max(axis=1)

# Debug: Display the first few rows of the new DataFrame
print("Updated DataFrame with AQI calculations:")
print(df_with_aqi.head())

# Display the first few rows of the updated dataset
df_with_aqi.head(5)

# Print column names
print(df_with_aqi.columns)

# Print column names as a list
column_names = df_with_aqi.columns.tolist()
print(column_names)

# Print column names with indices
for i, col in enumerate(df_with_aqi.columns):
    print(f"{i}: {col}")

# Print data types
print(df_with_aqi.dtypes)

# Print summary statistics
print(df_with_aqi.describe())

# Plot correlation heatmap
corr_matrix = df_with_aqi.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Plot overall AQI over time
df_with_aqi_reset = df_with_aqi.reset_index()
df_with_aqi_reset['date'] = pd.to_datetime(df_with_aqi_reset['date'])
df_with_aqi_reset.set_index('date', inplace=True)
df_with_aqi_reset['overall_aqi'].plot(figsize=(12, 6), title="Overall AQI Over Time")
plt.show()

# Train a RandomForestRegressor model
if 'overall_aqi' in df_with_aqi_reset.columns:
    if 'date' in df_with_aqi_reset.columns:
        X = df_with_aqi_reset.drop(columns=["overall_aqi", "date"])
    else:
        X = df_with_aqi_reset.drop(columns=["overall_aqi"])
    y = df_with_aqi_reset["overall_aqi"]
else:
    raise KeyError("Column 'overall_aqi' not found in DataFrame")

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

model = RandomForestRegressor()
model.fit(X_imputed, y)
importance = model.feature_importances_
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)
print(feature_importance)

# Split data into train, validation, and test sets
df_combined_reset = df_with_aqi.reset_index()
train = df_combined_reset[df_combined_reset["date"] < "2024-01-01"]
val = df_combined_reset[(df_combined_reset["date"] >= "2024-01-01") & (df_combined_reset["date"] < "2025-01-01")]
test = df_combined_reset[df_combined_reset["date"] >= "2025-01-01"]

# Standardize features
features = [
    'temperature_2m_max (°C)', 'temperature_2m_mean (°C)', 'precipitation_sum (mm)', 
    'wind_speed_10m_max (km/h)', 'wind_direction_10m_dominant (°)', 'shortwave_radiation_sum (MJ/m²)', 
    'pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'day_of_week', 'month', 'season', 
    'pm25_lag1', 'pm25_lag2', 'pm25_7day_avg', 'time_idx', 'pm25_aqi', 'pm10_aqi', 
    'o3_aqi', 'no2_aqi', 'so2_aqi', 'co_aqi', 'overall_aqi'
]

# Ensure the columns exist in the DataFrames
for feature in features:
    if feature not in train.columns:
        train[feature] = 0
    if feature not in val.columns:
        val[feature] = 0
    if feature not in test.columns:
        test[feature] = 0

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the scaler on the training data
train.loc[:, features] = scaler.fit_transform(train[features])

# Transform the validation and test data
val.loc[:, features] = scaler.transform(val[features])
test.loc[:, features] = scaler.transform(test[features]) 
 
here is the code now i want to use tft transformenr after this what all should i do and what all ca i predict based on above
To use the Temporal Fusion Transformer (TFT) after the preprocessing steps you've already completed, you'll need to follow these steps:

1. Install the Necessary Libraries
You need to install the pytorch-forecasting library, which provides the TFT model.

You can install it using pip:

bash
Copy
pip install pytorch-forecasting
2. Prepare the Data for TFT
TFT requires the data to be in a specific format, typically a TimeSeriesDataSet object.

You need to define the time index, target variable, and other relevant features.

3. Define the TimeSeriesDataSet
Create a TimeSeriesDataSet object from your preprocessed data.

Specify the time index, target variable, and other features.

4. Train the TFT Model
Initialize the TFT model with appropriate hyperparameters.

Train the model using the TimeSeriesDataSet.

5. Make Predictions
Use the trained TFT model to make predictions on the validation and test sets.

6. Evaluate the Model
Evaluate the model's performance using metrics like MAE, RMSE, etc.

7. Visualize the Results
Visualize the predictions against the actual values to understand the model's performance.

Step-by-Step Implementation
1. Install the Necessary Libraries
bash
Copy
pip install pytorch-forecasting
2. Prepare the Data for TFT
Ensure your data is in the correct format. You already have a time index (time_idx), target variable (overall_aqi), and other features.

3. Define the TimeSeriesDataSet
python
Copy
from pytorch_forecasting import TimeSeriesDataSet

# Define the TimeSeriesDataSet
max_prediction_length = 30  # Example: predict the next 30 days
max_encoder_length = 365  # Example: use the past 365 days as context

training = TimeSeriesDataSet(
    train,
    time_idx="time_idx",
    target="overall_aqi",
    group_ids=["time_idx"],  # Assuming no groups, but you can specify if needed
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],  # Specify any static categorical features
    static_reals=[],  # Specify any static real-valued features
    time_varying_known_categoricals=["day_of_week", "month", "season"],  # Specify time-varying known categorical features
    time_varying_known_reals=["temperature_2m_max (°C)", "temperature_2m_mean (°C)", "precipitation_sum (mm)", 
                              "wind_speed_10m_max (km/h)", "wind_direction_10m_dominant (°)", "shortwave_radiation_sum (MJ/m²)", 
                              "pm25", "pm10", "o3", "no2", "so2", "co", "pm25_lag1", "pm25_lag2", "pm25_7day_avg"],  # Specify time-varying known real-valued features
    time_varying_unknown_categoricals=[],  # Specify time-varying unknown categorical features
    time_varying_unknown_reals=["overall_aqi"],  # Specify time-varying unknown real-valued features
    target_normalizer=None,  # You can specify a normalizer if needed
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# Create validation dataset
validation = TimeSeriesDataSet.from_dataset(training, val, predict=True, stop_randomization=True)
4. Train the TFT Model
python
Copy
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

# Initialize the TFT model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

# Create a PyTorch Lightning Trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
trainer = Trainer(
    max_epochs=50,
    gpus=1,  # Use GPU if available
    gradient_clip_val=0.1,
    limit_train_batches=30,
    callbacks=[early_stop_callback],
)

# Fit the model
trainer.fit(
    tft,
    train_dataloader=training.to_dataloader(train=True, batch_size=128, num_workers=0),
    val_dataloader=validation.to_dataloader(train=False, batch_size=128, num_workers=0),
)
5. Make Predictions
python
Copy
# Make predictions on the validation set
predictions = tft.predict(validation.to_dataloader(train=False, batch_size=128, num_workers=0))

# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(predictions.numpy(), columns=[f"pred_{i}" for i in range(predictions.shape[1])])
6. Evaluate the Model
python
Copy
from pytorch_forecasting.metrics import MAE, RMSE

# Calculate MAE and RMSE
mae = MAE()(predictions, val["overall_aqi"].values)
rmse = RMSE()(predictions, val["overall_aqi"].values)

print(f"MAE: {mae}, RMSE: {rmse}")
7. Visualize the Results
python
Copy
import matplotlib.pyplot as plt

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(val["date"], val["overall_aqi"], label="Actual")
plt.plot(val["date"], predictions_df["pred_0"], label="Predicted")
plt.legend()
plt.title("Actual vs Predicted AQI")
plt.show()

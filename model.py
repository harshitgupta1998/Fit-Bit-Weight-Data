import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import numpy as np
def model():
    data1 = pd.read_csv('weightLogInfo_merged.csv')
    data2 = pd.read_csv('dailyActivity_merged.csv')
    data= pd.merge(data1, data2, on=['Date', 'Id'])
    data = data.drop(['Date', 'IsManualReport', 'LogId', 'Id', 'ActivityDate'], axis=1)
    data = data.fillna(0)
    X = data[['VeryActiveDistance', 'ModeratelyActiveDistance', 'LightActiveDistance', 'SedentaryActiveDistance', 'VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes', 'Calories']]
    y_weight = data['WeightKg']
    y_bmi = data['BMI']

    # Split the data into training and testing sets
    X_train, X_test, y_weight_train, y_weight_test, y_bmi_train, y_bmi_test = train_test_split(X, y_weight, y_bmi, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create linear regression models for weight and BMI
    model_weight = LinearRegression()
    model_bmi = LinearRegression()

    # Train the models
    model_weight.fit(X_train_scaled, y_weight_train)
    model_bmi.fit(X_train_scaled, y_bmi_train)

    # Make predictions
    weight_predictions = model_weight.predict(X_test_scaled)
    bmi_predictions = model_bmi.predict(X_test_scaled)

    # Evaluate the models
    mse_weight = mean_squared_error(y_weight_test, weight_predictions)
    mse_bmi = mean_squared_error(y_bmi_test, bmi_predictions)

    print(f'Mean Squared Error (Weight): {mse_weight}')
    print(f'Mean Squared Error (BMI): {mse_bmi}')

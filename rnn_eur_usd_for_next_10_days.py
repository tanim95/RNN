# -*- coding: utf-8 -*-
"""RNN_EUR/USD for Next 10 Days.ipynb

Original file is located at
    https://colab.research.google.com/drive/1QOnzIs9d3J7BZgbbcP2xjKd-mAtATm7E
"""

import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.optimizers import Adam
import math

df = pd.read_csv('EURUSD=X.csv')
df.tail()

df.describe()

df.dropna(inplace=True)

plt.figure(figsize=(14, 6), dpi=100)
plt.plot(df['Close'], label='Closing Price', color='blue')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('EURUSD Stock Closing Price')
plt.legend()
plt.grid(True)
plt.show()

df.set_index('Date', inplace=True)
data = df['Close'].values.reshape(-1, 1)
data

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

sequence_length = 10

X, y = [], []

for i in range(len(data) - sequence_length):
    X.append(data[i:i + sequence_length])
    y.append(data[i + sequence_length])

X = np.array(X)
y = np.array(y)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(
    X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2)

y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)
y_pred[:10, :]

mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(y_pred, label='Predicted Prices', color='red')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('EURUSD Price Prediction')
plt.legend()
plt.grid(True)
plt.show()


start_date = datetime.datetime(2024, 2, 23)

# Generating dates for the next 10 days
future_dates = [start_date + datetime.timedelta(days=i) for i in range(10)]

# Converting the dates to strings
next_dates = [date.strftime('%Y-%m-%d') for date in future_dates]
future_days = len(future_dates)
# Reshaping the data for prediction
X_future = X_test[-future_days:]

# Predicting the closing prices for the next 10 days
y_pred = model.predict(X_future)
y_pred = scaler.inverse_transform(y_pred).flatten()

# DataFrame with the predicted closing prices and dates
prediction_df = pd.DataFrame({'Date': next_dates, 'Predicted Price': y_pred})
print(prediction_df)

plt.figure(figsize=(10, 6))
plt.plot(prediction_df['Date'], prediction_df['Predicted Price'],
         label='Predicted Prices', color='red', marker='x')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Predicted Closing Prices of EUR/USD for Next 10 Days')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

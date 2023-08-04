
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

df = pd.read_csv('./data/GOOG.csv')
df.head()

# plt.figure(figsize=(10, 6), dpi=100)
# plt.plot(df['Close'], label='Closing Price', color='blue')
# plt.xlabel('Date')
# plt.ylabel('Closing Price')
# plt.title('Google Stock Closing Price')
# plt.legend()
# plt.grid(True)
# plt.show()

df.set_index('Date', inplace=True)
data = df['Close'].values.reshape(-1, 1)
data

"""## OR WE CAN USE MULTIPLE FEATURE COLUMN"""

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

sequence_length = 20

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
model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.2)

y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# Model Evaluation
loss = model.evaluate(X_test, y_test)

# Model's performance
print(f'Test Loss: {loss}')


plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(y_pred, label='Predicted Prices', color='red')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Google Stock Price Prediction')
plt.legend()
plt.grid(True)
plt.show()

"""## If we consider all the necessary feature not just Closing price"""

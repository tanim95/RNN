from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


data = pd.read_csv('World-Stock-Prices-Dataset.csv')
data.head(10)

data.isnull().sum()


data.columns

l_enc = LabelEncoder()
data['Brand_Name'] = l_enc.fit_transform(data['Brand_Name'])
data['Ticker'] = l_enc.fit_transform(data['Ticker'])
data['Industry_Tag'] = l_enc.fit_transform(data['Industry_Tag'])
data['Country'] = l_enc.fit_transform(data['Country'])
data.head(10)

scaler = StandardScaler()
numaric_cols = ['Open', 'High', 'Low', 'Close',
                'Volume', 'Dividends', 'Stock Splits']
data[numaric_cols] = scaler.fit_transform(data[numaric_cols])
data.head(10)

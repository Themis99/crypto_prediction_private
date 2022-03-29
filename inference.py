import tensorflow as tf
from tensorflow import keras
import requests
import yfinance as yf
import pandas as pd
import numpy as np



# required for solving issue with cuDNN
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

tf.random.set_seed(42)

def _request_data(days = 24):

    r = requests.get('http://api.alternative.me/fng/?limit=0')
    df = pd.DataFrame(r.json()['data'])
    df.value = df.value.astype(int)
    df.timestamp = pd.to_datetime(df.timestamp, unit='s')
    df.set_index('timestamp', inplace=True)
    df = df[::-1]
    df_btc = yf.download('BTC-USD')
    df_btc.index.name = 'timestamp'
    merged = df.merge(df_btc, on='timestamp')
    merged = merged.drop(['value_classification', 'time_until_update'], axis=1)
    merged = merged[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'value']]

    return merged.tail(days)


def _data_transformation():
    #request data
    data_f = _request_data()

    # data preprocessing
    # check for missing values
    data_f.isnull().any()

    # std normalization
    timeseries = data_f
    timeseries_mean = timeseries.mean(axis=0)
    timeseries_std = timeseries.std(axis=0)
    timeseries -= timeseries_mean
    timeseries /= timeseries_std
    timeseries = timeseries.values

    # to array
    data_array = np.array([timeseries])

    return data_array, timeseries_mean, timeseries_std

def _predict(PATH):
    data_p, m, s = _data_transformation()
    data_k = _request_data()
    # load model
    model = keras.models.load_model(PATH)
    # predict
    predict = model.predict(data_p)
    # de normalize
    real_value = predict[0][0] * s[3] + m[3]

    if real_value > data_k.tail(1)['Close'][0]:
        signal = "up"
    elif real_value < data_k.tail(1)['Close'][0]:
        signal = "down"

    return real_value, signal










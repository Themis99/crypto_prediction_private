import tensorflow as tf
from tensorflow import keras
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from id_stats import rolling_zscore
from tcn import TCN, tcn_full_summary
from datetime import datetime

class predictor:
    def __init__(self,days):
        self.days = days
        self.prediction_day = None
        tf.random.set_seed(42)

    def request_data(self):

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
        print(merged['Close'])
        pred_day = merged.index[-1]
        self.prediction_day = str((pred_day.strftime("%Y-%m-%d")))
        merged = merged[:-1]

        return merged.tail(2*self.days+1)


    def data_transformation(self):
        #request data
        data_f = self.request_data()

        # data preprocessing
        # check for missing values
        data_f.isnull().any()

        roll_z = rolling_zscore(window=self.days)
        data_scaled = roll_z.fit(data_f)

        return data_scaled, roll_z.m['Close'], roll_z.s['Close']

    def predict(self, PATH):
        data_p, m, s = self.data_transformation()
        data_k = self.request_data()

        data_k.reset_index(inplace=True)
        data_k = data_k.iloc[:, 1:]  # no date

        # load model
        data_p = data_p.tail(self.days)
        data_p = data_p.values
        data_p = data_p.reshape(1,self.days,7)
        model = keras.models.load_model(PATH)
        # predict
        tf.random.set_seed(42)
        predict = model.predict(data_p)

        # de-normalize
        last = data_k.tail(self.days-1)
        mean = last['Close'].mean()
        std = last['Close'].std()
        predict = predict[0][0]*std+mean

        #take the previous day
        previous_day = data_k.tail(1)
        previous_close = previous_day['Close'].item()


        if predict > previous_close:
            signal = 'UP'
            return signal, predict, self.prediction_day, previous_close
        elif predict < previous_close:
            signal = 'DOWN'
            return signal, predict, self.prediction_day,previous_close









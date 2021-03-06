import eth_data_collector
from eth_rolling import rolling_zscore
from tcn import TCN, tcn_full_summary
import tensorflow as tf
from tensorflow import keras
from datetime import date

tf.random.set_seed(42)

class predictor:
    def __init__(self,LAG):
        self.LAG = LAG


    def take_data(self):

        #retrieve data
        data = data_collector.yahoo_retriever()
        data = data[:-1]
        data = data.tail(self.LAG*2)
        return data

    def preprocess(self):

        #preprocess data
        data = self.take_data()
        # rolling z-score
        roll_z = rolling_zscore(window=self.LAG)
        data_scaled = roll_z.fit(data)
        data_scaled = data_scaled.tail(self.LAG)

        return data_scaled

    def predict(self,PATH):

        #take initial data
        data_init = self.take_data()
        #take previous close date
        prev_date = data_init.index[-1]
        previous_date = str((prev_date.strftime("%Y-%m-%d")))


        ################# edw exoume 8ema ###############
        #take prediction date
        data_alt = data_collector.yahoo_retriever()
        pred_date = data_alt.index[-1]
        prediction_date = str((pred_date.strftime("%Y-%m-%d")))
        #######################################

        #no date
        data_init.reset_index(inplace=True)
        data_init = data_init.iloc[:, 1:]  # no date

        #load model
        model = keras.models.load_model(PATH)

        #predict
        data_preprocessed = self.preprocess()
        data_preprocessed = data_preprocessed.values
        data_preprocessed = data_preprocessed.reshape(1, self.LAG, 6)
        predict = model.predict(data_preprocessed)

        # invert z-score
        last = data_init.tail(self.LAG-1)
        mean = last['Close'].mean()
        std = last['Close'].std()
        predict = predict[0][0] * std + mean
        predict = round(predict, 2)

        # take the previous day
        previous_day = data_init.tail(1)
        previous_close = previous_day['Close'].item()
        previous_close = round(previous_close, 2)

        if predict > previous_close:
            signal = 'UP'
            return signal, predict, prediction_date, previous_close,previous_date
        elif predict < previous_close:
            signal = 'DOWN'
            return signal, predict, prediction_date, previous_close, previous_date
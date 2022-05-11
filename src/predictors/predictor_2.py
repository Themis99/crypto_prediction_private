import tensorflow as tf
from tensorflow import keras

from src.data_collector import retrieve_data
from src.rolling import rolling_zscore

tf.random.set_seed(42)


class predictor_2:
    def __init__(self, LAG, past):
        self.LAG = LAG
        self.past = past

    def take_data(self):

        # retrieve data
        data = retrieve_data()
        data = data[:-self.past]
        data = data.tail(self.LAG * 2 + 1)
        return data

    def preprocess(self):

        # preprocess data
        data = self.take_data()
        # rolling z-score
        roll_z = rolling_zscore(window=self.LAG)
        data_scaled = roll_z.fit(data)
        data_scaled = data_scaled.tail(self.LAG)

        return data_scaled

    def predict(self, PATH):

        # take initial data
        data_init = self.take_data()

        # take previous close date
        prev_date = data_init.index[-1]
        previous_date = str((prev_date.strftime("%Y-%m-%d")))

        # take prediction date
        data_alt = retrieve_data()
        data_alt = data_alt[:-(self.past - 1)]
        # print(data_alt)
        pred_date = data_alt.index[-1]
        # print(pred_date)
        prediction_date = str((pred_date.strftime("%Y-%m-%d")))

        # no date
        data_init.reset_index(inplace=True)
        data_init = data_init.iloc[:, 1:]  # no date

        # load model
        model = keras.models.load_model(PATH)

        # predict
        data_preprocessed = self.preprocess()
        data_preprocessed = data_preprocessed.values
        data_preprocessed = data_preprocessed.reshape(1, self.LAG, 6)
        predict = model.predict(data_preprocessed)

        # invert z-score
        last = data_init.tail(self.LAG)
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
            return signal, predict, prediction_date, previous_close, previous_date
        elif predict < previous_close:
            signal = 'DOWN'
            return signal, predict, prediction_date, previous_close, previous_date

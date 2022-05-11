from tcn import TCN, tcn_full_summary
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras_tuner as kt
from rolling import rolling_zscore
import os
import data_collector
import time
import pickle

# required for solving issue with cuDNN
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))

tf.random.set_seed(42)

def plot_training_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.legend()
    plt.show()

# creating dataset
def create_sequences(timeseries, pred_var_index, p, h=1):
    (n, d) = timeseries.shape
    sequences = np.zeros((n - h - p, p, d)).astype('float32')
    targets = np.zeros((n - h - p)).astype('float32')
    for i in range(p, n - h):
        #print(i-p, i-1, i+h-1)
        sequence = timeseries[(i - p) : i, :]
        target = timeseries[i + h - 1, pred_var_index]
        sequences[i - p, :, :] = sequence
        targets[i - p] = target
    return sequences, targets

def def_model():
    model = Sequential()
    model.add(TCN(input_shape=(timesteps, input_dim)))
    model.add(Dense(1))

    return model

if __name__ == "__main__":
    # import data
    data = data_collector.yahoo_retriever()


    #rolling z-score
    lAG = 86
    roll_z = rolling_zscore(window=lAG)
    data_scaled = roll_z.fit(data)

    #dataset creation
    past = lAG
    horizon = 1
    pred_var_index = data.columns.get_loc('Close')
    data_values = data_scaled.values
    sequences, targets = create_sequences(data_values, pred_var_index, past, horizon)

    # train test split
    print('data shapes:', sequences.shape, targets.shape)

    rest = sequences.shape[0] - 1400

    X_train = sequences[0:1400,:,:]
    y_train = targets[0:1400]
    X_test = sequences[-rest:,:,:]
    y_test = targets[-rest:]

    batch_size = 2
    timesteps = lAG
    input_dim = len(data.columns)

    rmse = tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error", dtype=None)
    mse = tf.keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)
    mae = tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error", dtype=None)
    mape = tf.keras.metrics.MeanAbsolutePercentageError(name="mean_absolute_percentage_error", dtype=None)

    l = tf.keras.losses.LogCosh()
    learning_rate = 0.001

    model = def_model()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=l, metrics=[rmse, mse, mae, mape])
    history = model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_data=(X_test, y_test))

    plot_training_history(history)

    #evaluate on test set
    results = model.evaluate(X_test, y_test)

    print('TNC RMSE error (%): ', round(results[1], 3))
    print('TNC MSE error (%): ',round(results[2],3))
    print('TNC MAE error (%)): ',round(results[3],3))
    print('TNC MAPE error (%)): ',round(results[4],3))

    means_y = roll_z.m['Close']
    means_y = means_y.tail(y_test.shape[0] + 1).iloc[:-1].to_frame().reset_index(drop=True)

    std_y = roll_z.s['Close']
    std_y = std_y.tail(y_test.shape[0] + 1).iloc[:-1].to_frame().reset_index(drop=True)

    y_real = data['Close'].tail(y_test.shape[0] + 1).iloc[:-1].to_frame().reset_index(drop=True)

    # predict
    TNC_preds = model.predict(X_test).flatten(order='C')
    TNC_preds = pd.Series(TNC_preds)
    TNC_preds = TNC_preds * std_y['Close'] + means_y['Close']

    # plot real-predict
    plt.figure()
    plt.plot(y_real, label='Real')
    plt.plot(TNC_preds, label='Predictions')
    plt.legend()
    plt.show()
from tcn import TCN, tcn_full_summary
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
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

def build_model(hp):
    model = Sequential()
    NB_FILTERS = hp.Choice("nb_filters",[8,16,32,64,128])
    KERNEL_SIZE = hp.Choice("kernel_size", [2,3,4,8,12])
    NB_STACKS = hp.Choice("nb_stacks",[1,2,3,4])

    dropout_type = hp.Choice("dropout_type", ["dropout", "no_dropout"])
    if dropout_type == "dropout":
        with hp.conditional_scope("dropout_type",["dropout"]):
            dilations_s = hp.Choice("dilations_s", ["8", "16", "32", "64", "128"])
            if dilations_s == "8":
                with hp.conditional_scope("dilations_s", ["8"]):
                    model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8), dropout_rate=0.05, kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS))
            if dilations_s == "16":
                with hp.conditional_scope("dilations_s", ["16"]):
                    model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16), dropout_rate=0.05, kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS))
            if dilations_s == "32":
                with hp.conditional_scope("dilations_s", ["32"]):
                    model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32), dropout_rate=0.05, kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS))
            if dilations_s == "64":
                with hp.conditional_scope("dilations_s", ["64"]):
                    model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32, 64), dropout_rate=0.05, kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS))
            if dilations_s == "128":
                with hp.conditional_scope("dilations_s", ["128"]):
                    model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32, 64, 128), dropout_rate=0.05, kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS))
    if dropout_type == "no_dropout":
        with hp.conditional_scope("dropout_type",["no_dropout"]):
            dilations_s = hp.Choice("dilations_s", ["8", "16", "32", "64", "128"])
            if dilations_s == "8":
                with hp.conditional_scope("dilations_s", ["8"]):
                    model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8), kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS))
            if dilations_s == "16":
                with hp.conditional_scope("dilations_s", ["16"]):
                    model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16), kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS))
            if dilations_s == "32":
                with hp.conditional_scope("dilations_s", ["32"]):
                    model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32), kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS))
            if dilations_s == "64":
                with hp.conditional_scope("dilations_s", ["64"]):
                    model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32, 64), kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS))
            if dilations_s == "128":
                with hp.conditional_scope("dilations_s", ["128"]):
                    model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32, 64, 128), kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS))

    model.add(Dense(1))
    l = tf.keras.losses.LogCosh()
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=l, metrics=[rmse,mse,mae,mape])
    return model




if __name__ == "__main__":
    # import data
    data = data_collector.yahoo_retriever()
    print(data.head())

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

    #metrics
    rmse = tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error", dtype=None)
    mse = tf.keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)
    mae = tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error", dtype=None)
    mape = tf.keras.metrics.MeanAbsolutePercentageError(name="mean_absolute_percentage_error", dtype=None)

    #build model
    build_model(kt.HyperParameters())

    tuner = kt.BayesianOptimization(
        hypermodel=build_model,
        objective=kt.Objective("val_mean_absolute_percentage_error", direction="min"),
        max_trials=30,
        executions_per_trial=2,
        overwrite=True,
        directory="C:\\Users\\Themis\\Desktop\\TNC",
        project_name="TCN_bitcoin"
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=35)

    start = time.time()
    tuner.search(X_train, y_train, epochs=100, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[stop_early])
    end = time.time()
    print('time: '+ str(end - start))


    best_model = tuner.get_best_models()[0]

    #evaluate on test set
    results = best_model.evaluate(X_test, y_test)

    print('TNC RMSE error (%): ', round(results[1], 3))
    print('TNC MSE error (%): ',round(results[2],3))
    print('TNC MAE error (%)): ',round(results[3],3))
    print('TNC MAPE error (%)): ',round(results[4],3))

    means_y = roll_z.m['Close']
    means_y = means_y.tail(y_test.shape[0] + 1).iloc[:-1].to_frame().reset_index(drop=True)

    std_y = roll_z.s['Close']
    std_y = std_y.tail(y_test.shape[0] + 1).iloc[:-1].to_frame().reset_index(drop=True)

    y_real = data['Close'].tail(y_test.shape[0] + 1).iloc[:-1].to_frame().reset_index(drop=True)

    #predict
    TNC_preds = best_model.predict(X_test).flatten(order='C')
    TNC_preds = pd.Series(TNC_preds)
    TNC_preds = TNC_preds * std_y['Close'] + means_y['Close']

    #plot real-predict
    plt.figure()
    plt.plot(y_real, label='Real')
    plt.plot(TNC_preds, label='Predictions')
    plt.legend()
    plt.show()

    #save object of best hyperparameters
    best_hp = tuner.get_best_hyperparameters()
    with open('best_hp', 'wb') as config_dictionary_file:
        pickle.dump(best_hp, config_dictionary_file)

    best_model.save('C:\\Users\\Themis\\Desktop\\Cuda_bitcoin_pred\\etherium\\model\\eth_model_1')
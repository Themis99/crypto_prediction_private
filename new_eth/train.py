from tcn import TCN, tcn_full_summary
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import keras_tuner as kt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import data_collector
import matplotlib.pyplot as plt
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
    data = data.reset_index(drop=True)
    # take column names

    # train-test split
    days_for_test = 90  # days for testing

    X = data.values
    y = data['Close'].values
    X_train = X[0:(data.shape[0] - days_for_test)]
    X_test = X[-days_for_test:]
    y_train = y[0:(data.shape[0] - days_for_test)]
    y_test = y[-days_for_test:]

    # scaling
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    data_scaled = np.concatenate((X_train_scaled, X_test_scaled), axis=0)

    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train.reshape(-1, 1))

    # dataset creation
    past = 74
    horizon = 1
    pred_var_index = data.columns.get_loc('Close')
    sequences, targets = create_sequences(data_scaled, pred_var_index, past, horizon)

    # train test split of the sequence data
    print('data shapes:', sequences.shape, targets.shape)

    X_train = sequences[0:(data.shape[0] - days_for_test), :, :]
    y_train = targets[0:(data.shape[0] - days_for_test)]
    X_test = sequences[-days_for_test:, :, :]
    y_test = targets[-days_for_test:]

    batch_size = 1
    timesteps = 74
    input_dim = len(data.columns)

    rmse = tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error", dtype=None)
    mse = tf.keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)
    mae = tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error", dtype=None)
    mape = tf.keras.metrics.MeanAbsolutePercentageError(name="mean_absolute_percentage_error", dtype=None)

    #build model
    build_model(kt.HyperParameters())

    tuner = kt.BayesianOptimization(
        hypermodel=build_model,
        objective=kt.Objective("val_mean_absolute_percentage_error", direction="min"),
        max_trials=1,
        executions_per_trial=2,
        overwrite=True,
        directory="C:\\Users\\Themis\\Desktop\\TNC",
        project_name="TCN_bitcoin"
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=35)

    start = time.time()
    tuner.search(X_train, y_train, epochs=10, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[stop_early])
    end = time.time()
    print('time: '+ str(end - start))

    best_model = tuner.get_best_models()[0]

    #evaluate on test set
    results = best_model.evaluate(X_test, y_test)

    print('TNC RMSE error (%): ', round(results[1], 3))
    print('TNC MSE error (%): ',round(results[2],3))
    print('TNC MAE error (%)): ',round(results[3],3))
    print('TNC MAPE error (%)): ',round(results[4],3))

    #save best metrics
    metrics_ = [round(results[1], 3),round(results[2],3),round(results[3],3),round(results[4],3)]
    metrics_ = pd.Series(metrics_,index=['RMSE', 'MSE', 'MAE','MAPE'])
    metrics_.to_csv('./fine_tuning_performance.csv')

    y_real = data['Close'].tail(y_test.shape[0] + 1).iloc[:-1].to_frame().reset_index(drop=True)

    # predict
    preds = best_model.predict(X_test)
    preds_rescaled = scaler_y.inverse_transform(preds)
    scaler_params = {'Max': scaler_y.data_max_,'Min': scaler_y.data_min_}

    #save scaler parameteres
    with open('./saved_dictionary.pkl', 'wb') as f:
        pickle.dump(scaler_params, f)

    # plot real-predict
    plt.figure()
    plt.plot(y_real, label='Real')
    plt.plot(preds_rescaled, label='Predictions')
    plt.legend()
    plt.show()

    # save object of best hyperparameters
    best_hp = tuner.get_best_hyperparameters()
    with open('./best_hp', 'wb') as config_dictionary_file:
        pickle.dump(best_hp, config_dictionary_file)

    best_model.save('./eth_model_1')

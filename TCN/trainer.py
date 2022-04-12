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
from id_stats import rolling_zscore
import os

# required for solving issue with cuDNN
os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(tf.__version__)

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
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=tf.keras.losses.LogCosh(), metrics=[rmse,mse,mae,mape])
    return model




if __name__ == "__main__":
    # import data
    data = pd.read_csv('C:\\Users\\Themis\\Desktop\\bitcoin_pred\\TCN\\datasets\\old_data.csv')
    data = data.iloc[:, 1:]  # no date
    print(data.head())

    #rolling z-score
    lAG = 24
    roll_z = rolling_zscore(window=lAG)
    data_scaled = roll_z.fit(data)

    #dataset creation
    past = lAG
    horizon = 1
    pred_var_index = data.columns.get_loc('Close')
    data_values = data_scaled.values
    sequences, targets = create_sequences(data_values, pred_var_index, past, horizon)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=0.2, shuffle=False)
    print('train shapes:', X_train.shape, y_train.shape)
    print('test_shapes:', X_test.shape, y_test.shape)

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
        max_trials=2,
        executions_per_trial=2,
        overwrite=True,
        directory="C:\\Users\\Themis\\Desktop\\TNC",
        project_name="TCN_bitcoin"
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(X_train, y_train, epochs=10, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[stop_early])

    best_model = tuner.get_best_models()[0]
    results = best_model.evaluate(X_test, y_test)

    print('TNC RMSE error (%): ', round(results[1], 3))
    print('TNC MSE error (%): ',round(results[2],3))
    print('TNC MAE error (%)): ',round(results[3],3))
    print('TNC MAPE error (%)): ',round(results[4],3))

    #print predictions
    TNC_preds = best_model.predict(X_test)

    means_y = roll_z.m['Close']
    means_y = means_y.tail(y_test.shape[0] + 1)
    means_y = means_y.iloc[:-1]
    means_y = means_y.to_frame()
    means_y = means_y.reset_index(drop=True)

    std_y = roll_z.s['Close']
    std_y = std_y.tail(y_test.shape[0] + 1)
    std_y = std_y.iloc[:-1]
    std_y = std_y.to_frame()
    std_y = std_y.reset_index(drop=True)

    y_real = data['Close'].tail(y_test.shape[0] + 1)
    y_real = y_real.iloc[:-1]
    y_real = y_real.to_frame()
    y_real = y_real.reset_index(drop=True)

    TNC_preds = TNC_preds.flatten(order='C')
    TNC_preds = pd.Series(TNC_preds)
    TNC_preds = TNC_preds * std_y['Close'] + means_y['Close']

    plt.figure()
    plt.plot(y_real, label='Real')
    plt.plot(TNC_preds, label='Predictions')
    plt.legend()
    plt.show()

    #save model
    #model.save('C:\\Users\\Themis\\Desktop\\bitcoin_pred\\Best models\\new_model')






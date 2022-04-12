from tcn import TCN
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler,StandardScaler


# required for solving issue with cuDNN
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

tf.random.set_seed(42)



# import data
data = pd.read_csv('C:\\Users\\Themis\\Desktop\\bitcoin_pred\\TCN\\datasets\\final_df.csv')
print(data.head())

# data preprocessing
# check for missing values
data.isnull().any()

timeseries = data.iloc[:, 1:] # no date
#std normalization
scaler = RobustScaler()
scaled_data = timeseries.copy()
scaled_data[timeseries.columns] = scaler.fit_transform(timeseries[timeseries.columns])

scaler = StandardScaler()
scaled_data[timeseries.columns] = scaler.fit_transform(scaled_data[timeseries.columns])

scaled_data = scaled_data.values

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



past = 24
horizon = 1
pred_var_index = 40

sequences, targets = create_sequences(scaled_data, pred_var_index, past, horizon)
sequences = sequences[:,:,:-1]

# train test split
X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=0.2, shuffle=False)

print('train shapes:', X_train.shape, y_train.shape)
print('test_shapes:',X_test.shape,y_test.shape)

batch_size = 2

timesteps = past

input_dim = 40

#loss
l = tf.keras.losses.LogCosh()
#compile and fit
rmse = tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error", dtype=None)
mse = tf.keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)
mae = tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error", dtype=None)
mape = tf.keras.metrics.MeanAbsolutePercentageError(name="mean_absolute_percentage_error", dtype=None)

def build_model(hp):
    model = Sequential()
    NB_FILTERS = hp.Choice("nb_filters",[8,16,32,64,128])
    KERNEL_SIZE = hp.Choice("kernel_size", [2,3,4,8,12])
    NB_STACKS = hp.Choice("nb_stacks",[1,2,3,4])

    normalization_type = hp.Choice("normalization_type", ['no_norm','use_batch_norm','use_layer_norm','use_weight_norm'])
    if normalization_type == 'no_norm':
        with hp.conditional_scope("normalization_type", ["no_norm"]):
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
    if normalization_type == 'use_batch_norm':
        with hp.conditional_scope("normalization_type", ["use_batch_norm"]):
            dropout_type = hp.Choice("dropout_type", ["dropout", "no_dropout"])
            if dropout_type == "dropout":
                with hp.conditional_scope("dropout_type",["dropout"]):
                    dilations_s = hp.Choice("dilations_s", ["8", "16", "32", "64", "128"])
                    if dilations_s == "8":
                        with hp.conditional_scope("dilations_s", ["8"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8), dropout_rate=0.05, kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_batch_norm = True))
                    if dilations_s == "16":
                        with hp.conditional_scope("dilations_s", ["16"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16), dropout_rate=0.05, kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_batch_norm = True))
                    if dilations_s == "32":
                        with hp.conditional_scope("dilations_s", ["32"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32), dropout_rate=0.05, kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_batch_norm = True))
                    if dilations_s == "64":
                        with hp.conditional_scope("dilations_s", ["64"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32, 64), dropout_rate=0.05, kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_batch_norm = True))
                    if dilations_s == "128":
                        with hp.conditional_scope("dilations_s", ["128"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32, 64, 128), dropout_rate=0.05, kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_batch_norm = True))
            if dropout_type == "no_dropout":
                with hp.conditional_scope("dropout_type",["no_dropout"]):
                    dilations_s = hp.Choice("dilations_s", ["8", "16", "32", "64", "128"])
                    if dilations_s == "8":
                        with hp.conditional_scope("dilations_s", ["8"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8), kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_batch_norm = True))
                    if dilations_s == "16":
                        with hp.conditional_scope("dilations_s", ["16"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16), kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_batch_norm = True))
                    if dilations_s == "32":
                        with hp.conditional_scope("dilations_s", ["32"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32), kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_batch_norm = True))
                    if dilations_s == "64":
                        with hp.conditional_scope("dilations_s", ["64"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32, 64), kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_batch_norm = True))
                    if dilations_s == "128":
                        with hp.conditional_scope("dilations_s", ["128"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32, 64, 128), kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_batch_norm = True))
    if normalization_type == 'use_layer_norm':
        with hp.conditional_scope("normalization_type", ["use_layer_norm"]):
            dropout_type = hp.Choice("dropout_type", ["dropout", "no_dropout"])
            if dropout_type == "dropout":
                with hp.conditional_scope("dropout_type",["dropout"]):
                    dilations_s = hp.Choice("dilations_s", ["8", "16", "32", "64", "128"])
                    if dilations_s == "8":
                        with hp.conditional_scope("dilations_s", ["8"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8), dropout_rate=0.05, kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_layer_norm = True))
                    if dilations_s == "16":
                        with hp.conditional_scope("dilations_s", ["16"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16), dropout_rate=0.05, kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_layer_norm = True))
                    if dilations_s == "32":
                        with hp.conditional_scope("dilations_s", ["32"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32), dropout_rate=0.05, kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_layer_norm = True))
                    if dilations_s == "64":
                        with hp.conditional_scope("dilations_s", ["64"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32, 64), dropout_rate=0.05, kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_layer_norm = True))
                    if dilations_s == "128":
                        with hp.conditional_scope("dilations_s", ["128"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32, 64, 128), dropout_rate=0.05, kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_layer_norm = True))
            if dropout_type == "no_dropout":
                with hp.conditional_scope("dropout_type",["no_dropout"]):
                    dilations_s = hp.Choice("dilations_s", ["8", "16", "32", "64", "128"])
                    if dilations_s == "8":
                        with hp.conditional_scope("dilations_s", ["8"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8), kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_layer_norm = True))
                    if dilations_s == "16":
                        with hp.conditional_scope("dilations_s", ["16"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16), kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_layer_norm = True))
                    if dilations_s == "32":
                        with hp.conditional_scope("dilations_s", ["32"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32), kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_layer_norm = True))
                    if dilations_s == "64":
                        with hp.conditional_scope("dilations_s", ["64"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32, 64), kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_layer_norm = True))
                    if dilations_s == "128":
                        with hp.conditional_scope("dilations_s", ["128"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32, 64, 128), kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_layer_norm = True))
    if normalization_type == 'use_weight_norm':
        with hp.conditional_scope("normalization_type", ["use_weight_norm"]):
            dropout_type = hp.Choice("dropout_type", ["dropout", "no_dropout"])
            if dropout_type == "dropout":
                with hp.conditional_scope("dropout_type",["dropout"]):
                    dilations_s = hp.Choice("dilations_s", ["8", "16", "32", "64", "128"])
                    if dilations_s == "8":
                        with hp.conditional_scope("dilations_s", ["8"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8), dropout_rate=0.05, kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_weight_norm = True))
                    if dilations_s == "16":
                        with hp.conditional_scope("dilations_s", ["16"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16), dropout_rate=0.05, kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_weight_norm = True))
                    if dilations_s == "32":
                        with hp.conditional_scope("dilations_s", ["32"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32), dropout_rate=0.05, kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_weight_norm = True))
                    if dilations_s == "64":
                        with hp.conditional_scope("dilations_s", ["64"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32, 64), dropout_rate=0.05, kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_weight_norm = True))
                    if dilations_s == "128":
                        with hp.conditional_scope("dilations_s", ["128"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32, 64, 128), dropout_rate=0.05, kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_weight_norm = True))
            if dropout_type == "no_dropout":
                with hp.conditional_scope("dropout_type",["no_dropout"]):
                    dilations_s = hp.Choice("dilations_s", ["8", "16", "32", "64", "128"])
                    if dilations_s == "8":
                        with hp.conditional_scope("dilations_s", ["8"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8), kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_weight_norm = True))
                    if dilations_s == "16":
                        with hp.conditional_scope("dilations_s", ["16"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16), kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_weight_norm = True))
                    if dilations_s == "32":
                        with hp.conditional_scope("dilations_s", ["32"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32), kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_weight_norm = True))
                    if dilations_s == "64":
                        with hp.conditional_scope("dilations_s", ["64"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32, 64), kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_weight_norm = True))
                    if dilations_s == "128":
                        with hp.conditional_scope("dilations_s", ["128"]):
                            model.add(TCN(input_shape=(timesteps, input_dim), nb_filters=NB_FILTERS, dilations=(1, 2, 4, 8, 16, 32, 64, 128), kernel_size=KERNEL_SIZE, nb_stacks=NB_STACKS,use_weight_norm = True))

    model.add(Dense(1))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=l, metrics=[rmse,mse,mae,mape])
    return model

build_model(kt.HyperParameters())

tuner = kt.BayesianOptimization(
    hypermodel=build_model,
    objective=kt.Objective("val_mean_squared_error", direction="min"),
    max_trials=40,
    executions_per_trial=2,
    overwrite=True,
    directory="C:\\Users\\Themis\\Desktop\\TNC",
    project_name="TCN_bitcoin"
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15)
tuner.search(X_train, y_train, epochs=100, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[stop_early])


best_model = tuner.get_best_models()[0]
results = best_model.evaluate(X_test, y_test)

print('TNC RMSE error (%): ', round(results[1], 3))
print('TNC MSE error (%): ',round(results[2],3))
print('TNC MAE error (%)): ',round(results[3],3))
print('TNC MAPE error (%)): ',round(results[4],3))

TNC_preds = best_model.predict(X_test)

plt.figure()
plt.plot(y_test, label='Real')
plt.plot(TNC_preds, label='Predictions')
plt.legend()
plt.show()

# model save
best_model.save('C:\\Users\\Themis\\Desktop\\bitcoin_pred\\Best models\\new')


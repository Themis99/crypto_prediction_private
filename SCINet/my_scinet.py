from typing import Tuple
import os
import random
import pandas as pd
import numpy as np
from joblib import dump, load
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from SCINet import SCINet, StackedSCINet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler,StandardScaler

# required for solving issue with cuDNN
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

tf.random.set_seed(42)


# Make model
def make_model(input_shape, output_shape):
    inputs = tf.keras.Input(shape=(input_shape[0], input_shape[1],input_shape[2]), name='inputs')
    # x = SciNet(horizon, levels=L, h=h, kernel_size=kernel_size)(inputs)
    # model = tf.keras.Model(inputs, x)
    targets = tf.keras.Input(shape=(output_shape[0]), name='targets')
    predictions = StackedSCINet(horizon=horizon, features=input_shape[-1], stacks=K, levels=L, h=h, kernel_size=kernel_size)(inputs, targets)
    model = tf.keras.Model(inputs=[inputs, targets], outputs=predictions)

    model.summary()
    tf.keras.utils.plot_model(model, to_file='modelDiagram.png', show_shapes=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse',
                  metrics=['mean_squared_error', 'mean_absolute_error'])
    return model


# import data
data = pd.read_csv('C:\\Users\\Themis\\Desktop\\bitcoin_pred\\TCN\\datasets\\final_df.csv')

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

plt.plot(y_test, label='Real')
plt.show()

look_back_window, horizon = 24, 1
batch_size = 4
learning_rate = 9e-3
h, kernel_size, L, K = 4, 5, 3, 2
l1, l2 = 0.001, 0.1

model = make_model(X_train.shape, y_train.shape)

#loss
l = tf.keras.losses.LogCosh()
#compile and fit
rmse = tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error", dtype=None)
mse = tf.keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)
mae = tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error", dtype=None)
mape = tf.keras.metrics.MeanAbsolutePercentageError(name="mean_absolute_percentage_error", dtype=None)


model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=l, metrics=[rmse, mse, mae, mape])
history = model.fit(X_train, y_train, epochs=100, validation_data = (X_test, y_test),batch_size=batch_size)

#training history
plot_training_history(history)
plt.show()

#prediction results
results = model.evaluate(X_test, y_test)

print('TNC RMSE error (%): ', round(results[1], 3))
print('TNC MSE error (%): ',round(results[2],3))
print('TNC MAE error (%)): ',round(results[3],3))
print('TNC MAPE error (%)): ',round(results[4],3))

TNC_preds = model.predict(X_test)

plt.figure()
plt.plot(y_test, label='Real')
plt.plot(TNC_preds, label='Predictions')
plt.legend()
plt.show()

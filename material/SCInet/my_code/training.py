import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from SCINet import SCINet, StackedSCINet



# required for solving issue with cuDNN
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

tf.random.set_seed(42)

#import data
data = pd.read_csv('C:\\Users\\Themis\\Desktop\\bitcoin_prediction\\my_code\\dataset\\data.csv')
data.head()

# check for missing values
data.isnull().any()

#std normalization
timeseries = data.iloc[:, 1:]
timeseries_mean = timeseries.mean(axis=0)
timeseries_std = timeseries.std(axis=0)
timeseries -= timeseries_mean
timeseries /= timeseries_std

timeseries = timeseries.values

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



past = 50
horizon = 1
pred_var_index = 3

sequences, targets = create_sequences(timeseries, pred_var_index, past, horizon)

# train test split
X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=0.1, shuffle=False)

# creating the model
def make_model(input_shape, output_shape):
    inputs = tf.keras.Input(shape=(input_shape[1], input_shape[2]), name='inputs')
    # x = SciNet(horizon, levels=L, h=h, kernel_size=kernel_size)(inputs)
    # model = tf.keras.Model(inputs, x)
    targets = tf.keras.Input(shape=(output_shape[1], output_shape[2]), name='targets')
    predictions = StackedSCINet(horizon=horizon, features=input_shape[-1], stacks=K, levels=L, h=h,
                                kernel_size=kernel_size,
                                regularizer=(l1, l2))(inputs, targets)
    model = tf.keras.Model(inputs=[inputs, targets], outputs=predictions)

    model.summary()
    tf.keras.utils.plot_model(model, to_file='modelDiagram.png', show_shapes=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse',
                  metrics=['mean_squared_error', 'mean_absolute_error'])
    return model

def plot_training_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.legend()
    plt.show()

# parameters
batch_size = 16
learning_rate = 9e-3
h, kernel_size, L, K = 4, 5, 3, 2
l1, l2 = 0.001, 0.1

model = make_model(X_train.shape, y_train.shape)
early_stopping = EarlyStopping(monitor='val_loss', patience=100, min_delta=0, verbose=1, restore_best_weights=True)
history = model.fit({'inputs': X_train, 'targets': y_train},
                        validation_data={'inputs': X_test, 'targets': y_test},
                        batch_size=batch_size, epochs=1600, callbacks=[early_stopping])

plot_training_history(history)
plt.show()

results = model.evaluate(X_test, y_test)

print('Feedforward test error (%): ', round(results[1], 3))

dnn_preds = model.predict(X_test)

plt.figure()
plt.plot(y_test, label='Real')
plt.plot(dnn_preds, label='Predictions')
plt.legend()
plt.show()
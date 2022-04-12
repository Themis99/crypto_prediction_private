# import data
from tcn import TCN, tcn_full_summary
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from id_stats import rolling_zscore
from sklearn.preprocessing import StandardScaler

def plot_training_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.legend()
    plt.show()

data = pd.read_csv('C:\\Users\\Themis\\Desktop\\bitcoin_pred\\TCN\\datasets\\old_data.csv')
data = data.iloc[:, 1:] # no date

#rolling z_score
#roll_z = rolling_zscore(window=24)
#data_scaled = roll_z.fit(data)
data_scaled = pd.DataFrame()
col = data.columns
scaler = StandardScaler()
data_scaled[col] = scaler.fit_transform(data[col])



#train test split
X = data_scaled
y = data_scaled['Close']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, shuffle=False)

print('train shapes:', X_train.shape, y_train.shape)
print('test_shapes:',X_test.shape,y_test.shape)

#creating samples
LAG = 24
FEATURES = 7
BATCH_SIZE_TRAIN = 2
BATCH_SIZE_TEST = 1
LR = 0.0001


gen_train = TimeseriesGenerator(X_train, y_train, length=LAG, sampling_rate=1, batch_size=BATCH_SIZE_TRAIN)
gen_test = TimeseriesGenerator(X_test, y_test, length=LAG, sampling_rate=1, batch_size=BATCH_SIZE_TEST)

print(len(gen_train))
print(len(gen_test))

# model creation
tcn = TCN(input_shape=(LAG, FEATURES))
model = Sequential([
    tcn,
    Dense(1)
])

#loss
l = tf.keras.losses.LogCosh()
#compile and fit
rmse = tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error", dtype=None)
mse = tf.keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)
mae = tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error", dtype=None)
mape = tf.keras.metrics.MeanAbsolutePercentageError(name="mean_absolute_percentage_error", dtype=None)

learning_rate = 0.001
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR), loss=l, metrics=[rmse, mse, mae, mape])

history = model.fit_generator(gen_train,steps_per_epoch=len(gen_train),epochs=10,validation_data = gen_test, validation_steps = len(gen_test),verbose= 1)


#training history
plot_training_history(history)
plt.show()

#prediction results
results = model.evaluate_generator(gen_test,steps = len(gen_test))

print('TNC RMSE error (%): ', round(results[1], 3))
print('TNC MSE error (%): ',round(results[2],3))
print('TNC MAE error (%)): ',round(results[3],3))
print('TNC MAPE error (%)): ',round(results[4],3))

TNC_preds = model.predict_generator(gen_test,steps = len(gen_test))

y_real = y_test[-280:]

plt.figure()
plt.plot(y_real, label='Real')
plt.plot(TNC_preds, label='Predictions')
plt.legend()
plt.show()







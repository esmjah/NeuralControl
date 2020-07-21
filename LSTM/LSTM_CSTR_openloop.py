import numpy as np
import numpy.random as npr
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from scipy.integrate import ode
keras = tf.keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
#import control as c
from scipy import io
from neural_ode_u import NeuralODE
import h5py
from numpy.random import seed
seed(0)
tf.random.set_seed(42)
tf.executing_eagerly()


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)


data = io.loadmat(file_name='cstr_data_1.mat')

dt = data['dt'][0][0]
z0 = data['z0'][0]
t_process = data['t_process'][0]
u_process = data['u_process']
y_process = data['y_process']


y1_process = [y_[0] for y_ in y_process]
y2_process = [y_[1] for y_ in y_process]

plt.close('all')
# plt.figure(1)
# plt.clf()
# plt.subplot(311)
# plt.plot(t_process,u_process,'b-',label='u(t)')
# plt.ylabel('values')
# plt.xlabel('time')
# plt.xlabel('time')
# plt.legend(loc='best')
# plt.subplot(312)
# plt.plot(t_process,y1_process,'b-',label='x(t)')
# plt.ylabel('values')
# plt.xlabel('time')
# plt.legend(loc='best')
# plt.subplot(313)
# plt.plot(t_process,y2_process,'b-',label='y(t)')
# plt.ylabel('values')
# plt.xlabel('time')
# plt.legend(loc='best')
# plt.show()

# scaler_u = MinMaxScaler((-1, 1))
# scaler_y = MinMaxScaler((-1, 1))
#
# scaler_u.fit(u_process)
# scaler_y.fit(y_process)
# u_train = scaler_u.transform(u_process)
# y_train = scaler_y.transform(y_process)

min_y = np.min(y_process, axis=0)
max_y = np.max(y_process, axis=0)
scale_y = 0.5*(max_y - min_y)

min_u = np.min(u_process, axis=0)
max_u = np.max(u_process, axis=0)
scale_u = 0.5*(max_u - min_u)

u_train = (u_process - u_process[0])/scale_u
y_train = (y_process - y_process[0])/scale_y
t_train_data = t_process


train_data = np.concatenate((y_train, u_train), axis=1)

col_= train_data.shape[1]
# load the test data
look_back = 5
no_of_features=3  # Ca, T ,u

trainX, trainY = create_dataset(train_data, look_back)
n_train = trainX.shape[0]

model = tf.keras.Sequential()
# model.add(LSTM(150, input_dim=look_back,return_sequences=True))
model.add(tf.keras.layers.LSTM(150, input_shape=(look_back, no_of_features), return_sequences=True))

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.LSTM(150, input_dim=look_back, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.LSTM(150, input_dim=look_back, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.LSTM(100, input_dim=look_back, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.LSTM(100, input_dim=look_back, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.LSTM(100, input_dim=look_back, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.LSTM(80, input_dim=look_back, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.LSTM(80, input_dim=look_back, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))

# model.add(LSTM(150, input_dim=look_back))
model.add(tf.keras.layers.LSTM(80, input_shape=(look_back, no_of_features)))
model.add(tf.keras.layers.Dropout(0.2))

model.add(Dense(col_))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=optimizer)

# make predictions
model.load_weights("LSTM_model_weights.ckpt")

# ############## Test on training data ################
predicted_y = model.predict(trainX)

loss_ = np.sum(np.square(predicted_y - train_data[look_back:,:]), axis=0)
training_loss = np.sum(loss_)
print("Training loss: ", training_loss)

plt.figure(3)
plt.suptitle("Training data")
plt.subplot(311)
plt.plot(t_train_data, u_train)
plt.subplot(312)
plt.plot(t_train_data, y_train[:, 0], '-*', markersize=2)
plt.plot(t_train_data[look_back:], np.array(predicted_y[:, 0]), '-o', markersize=2)
plt.subplot(313)
plt.plot(t_train_data, y_train[:, 1], '-*', markersize=2)
plt.plot(t_train_data[look_back:], np.array(predicted_y[:, 1]), '-o', markersize=2)


# ############## Test on bew data-set for model validation ################
t_test = data['t_test'][0]
u_test = data['u_test']
y_test = data['y_test']

# u_test_scaled = scaler_u.transform(u_test)
# y_test_scaled = scaler_y.transform(y_test)

u_test_scaled = (u_test - u_process[0])/scale_u
y_test_scaled = (y_test - y_process[0])/scale_y

test_data = np.concatenate((y_test_scaled, u_test_scaled), axis=1)
testX, testY = create_dataset(test_data, look_back)

y_test_predicted = model.predict(testX)
loss_ = np.sum(np.square(y_test_predicted - test_data[look_back:,:]), axis=0)
validation_loss = sum(loss_)

print("validation_loss:", validation_loss)

plt.figure(5)
plt.suptitle("Validation data")
plt.subplot(311)
plt.plot(t_test, u_test_scaled, '-*', markersize=2)
plt.plot(t_test[look_back:], np.array(y_test_predicted[:, 2]), '-o', markersize=2)
plt.subplot(312)
plt.plot(t_test, y_test_scaled[:, 0], '-*', markersize=2)
plt.plot(t_test[look_back:], np.array(y_test_predicted[:, 0]), '-o', markersize=2)
plt.subplot(313)
plt.plot(t_test, y_test_scaled[:, 1], '-*', markersize=2)
plt.plot(t_test[look_back:], np.array(y_test_predicted[:, 1]), '-o', markersize=2)

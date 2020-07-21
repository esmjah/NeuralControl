import numpy as np
import numpy.random as npr
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from scipy.integrate import ode
keras = tf.keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
#import control as c
from scipy import io
from neural_ode_u import NeuralODE
import h5py
tf.executing_eagerly()


def get_batch(x, start, batch_length):
    x_batch = x[int(start):int(start)+batch_length]
    return x_batch


data = io.loadmat(file_name='cstr_data_1.mat')

dt = data['dt'][0][0]
z0 = data['z0'][0]
t_process = data['t_process'][0]
u_process = data['u_process']
y_process = data['y_process']


y1_process = [y_[0] for y_ in y_process]
y2_process = [y_[1] for y_ in y_process]

plt.close('all')
plt.figure(1)
plt.clf()
plt.subplot(311)
plt.plot(t_process,u_process,'b-',label='u(t)')
plt.ylabel('values')
plt.xlabel('time')
plt.xlabel('time')
plt.legend(loc='best')
plt.subplot(312)
plt.plot(t_process,y1_process,'b-',label='x(t)')
plt.ylabel('values')
plt.xlabel('time')
plt.legend(loc='best')
plt.subplot(313)
plt.plot(t_process,y2_process,'b-',label='y(t)')
plt.ylabel('values')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()


min_y = np.min(y_process, axis=0)
max_y = np.max(y_process, axis=0)
scale_y = 0.5*(max_y - min_y)

min_u = np.min(u_process, axis=0)
max_u = np.max(u_process, axis=0)
scale_u = 0.5*(max_u - min_u)

u_train = (u_process - u_process[0])/scale_u
y_train = (y_process - y_process[0])/scale_y
t_train_data = t_process


u0 = tf.cast(u_train[0], dtype=tf.float32)
y0 = y_train[0]
y0_p = tf.cast([0, 0], dtype=tf.float32)

init_state = tf.concat([u0, y0, y0_p], axis=0)


class SOPTD(tf.keras.Model):
    def __init__(self):
        super(SOPTD, self).__init__()
        self.dense1 = keras.layers.Dense(4, input_shape=(1, 3), activation='linear', name='dense_1')
        self.dense2 = keras.layers.Dense(32, activation='tanh', name='dense_2')
        self.dense3 = keras.layers.Dense(32, activation='relu', name='dense_3')
        self.dense4 = keras.layers.Dense(2, activation='linear', name='dense_4')
        self.dense5 = keras.layers.Dense(32, input_shape=(1, 5), activation='linear', name='dense_5')
        self.dense6 = keras.layers.Dense(32, activation='tanh', name='dense_6')
        self.dense7 = keras.layers.Dense(32, activation='relu', name='dense_7')
        self.dense8 = keras.layers.Dense(4, activation='linear', name='dense_8')
        self.c1 = tf.Variable(tf.cast(1.0, dtype=tf.float32))
        self.c2 = tf.Variable(tf.cast(-1.0, dtype=tf.float32))
        self.a1 = tf.Variable(tf.cast(1.0, dtype=tf.float32))
        self.a2 = tf.Variable(tf.cast(1.0, dtype=tf.float32))
        self.b1 = tf.Variable(tf.cast(1.0, dtype=tf.float32))
        self.b2 = tf.Variable(tf.cast(1.0, dtype=tf.float32))

    def call(self, inputs, **kwargs):
        t, states = inputs
        u = states[0]
        x1 = states[1]
        x2 = states[2]
        x1_p = states[3]
        x2_p = states[4]
        u1 = tf.reshape([u, x1, x2], [1, 3])
        u2 = self.dense1(u1)
        u3 = self.dense2(u2)
        u4 = self.dense3(u3)
        u5 = self.dense4(u4)
        u6 = tf.reshape(u5, [2, ])
        x1_dot = x1_p
        x1_p_dot = self.c1*u6[0] - self.a1*x1 - self.b1*x1_p
        x2_dot = x2_p
        x2_p_dot = self.c2*u6[1] - self.a2*x2 - self.b2*x2_p
        x_dot = tf.concat([[x1_dot], [x2_dot], [x1_p_dot], [x2_p_dot], [u]], axis=0)
        g0 = tf.reshape(x_dot, [1, 5])
        g1 = self.dense5(g0)
        g2 = self.dense6(g1)
        g3 = self.dense7(g2)
        g4 = self.dense8(g3)
        g5 = tf.reshape(g4, [4, ])

        u_dot = tf.reshape(tf.cast(0, dtype=tf.float32), [1, ])
        h = tf.concat([u_dot, g5], axis=0)
        return h



model = SOPTD()
model.load_weights("CSTR_SecondOrder_MLP_v4_1.ckpt")

# ############## Test on training data ################
neural_ode_test = NeuralODE(model, t=t_train_data)
state_final, predicted_x = neural_ode_test.forward(init_state, u_train, return_states="numpy")
predicted_y = np.array(predicted_x[:,1:y_train.shape[1]+1])
loss_ = np.sum(np.square(predicted_y - y_train), axis=0)
training_loss = np.sum(loss_)
print("Training loss: ", training_loss)

plt.figure(3)
plt.suptitle("Training data")
plt.subplot(311)
plt.plot(t_train_data, u_train)
plt.subplot(312)
plt.plot(t_train_data, y_train[:, 0])
plt.plot(t_train_data, np.array(predicted_y[:, 0]))
plt.subplot(313)
plt.plot(t_train_data, y_train[:, 1])
plt.plot(t_train_data, np.array(predicted_y[:, 1]))


# ############## Test on bew data-set for model validation ################
t_test = data['t_test'][0]
u_test = data['u_test']
y_test = data['y_test']

# u_test_scaled = scaler_u.transform(u_test)
# y_test_scaled = scaler_y.transform(y_test)

u_test_scaled = (u_test - u_process[0])/scale_u
y_test_scaled = (y_test - y_process[0])/scale_y

u0_test = tf.cast(u_test_scaled[0], dtype=tf.float32)
y0_test = tf.reshape(tf.cast([y_test_scaled[0]], dtype=tf.float32), [2, ])

neural_ode_test = NeuralODE(model, t=t_test)
init_state_test = tf.concat([u0_test, y0_test, y0_p], axis=0)

state_final_test, predicted_x_test = neural_ode_test.forward(init_state_test, u_test_scaled, return_states="numpy")
y_test_predicted = np.array(predicted_x_test[:, 1:y_test.shape[1]+1])
loss_ = np.sum(np.square(y_test_predicted - y_test_scaled), axis=0)
validation_loss = sum(loss_)

print("validation_loss:", validation_loss)

plt.figure(5)
plt.suptitle("Validation data")
plt.subplot(311)
plt.plot(t_test, u_test_scaled)
plt.subplot(312)
plt.plot(t_test, y_test_scaled[:, 0])
plt.plot(t_test, np.array(y_test_predicted[:, 0]))
plt.subplot(313)
plt.plot(t_test, y_test_scaled[:, 1])
plt.plot(t_test, np.array(y_test_predicted[:, 1]))

# model.save_weights("CSTR_SecondOrder_MLP_v2.h5")

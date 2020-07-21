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
from numpy.random import seed
seed(0)
tf.random.set_seed(42)
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


# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)

model = SOPTD()
#for layer in model.layers: print(layer.get_config(), layer.get_weights())

batch_size = 100
t_sim = np.linspace(0, (batch_size - 1)*dt, num=batch_size)

neural_ode = NeuralODE(model, t=t_sim)


def compute_gradients_and_update_path(initial_state, u_target, y_target):
    with tf.GradientTape() as g:
        final_y, x_history = neural_ode.forward(initial_state, u_target, return_states="tf")
        predicted_path = x_history[:,1:y_target.shape[1]+1]  # -> (batch_time, batch_size, 2)
        loss_ = tf.reduce_sum(tf.square(predicted_path - y_target), axis=0)
        prediction_loss = tf.reduce_sum(loss_)

    # back-propagate through solver with tensorflow
    gradients = g.gradient(prediction_loss, model.weights)
    optimizer.apply_gradients(zip(gradients, model.weights))
    return prediction_loss, x_history

# Compile EAGER graph to static (this will be much faster)
compute_gradients_and_update = tf.function(compute_gradients_and_update_path)

# ############## Training ################
n_iters = 10000
idx = np.arange(len(y_train) - batch_size, dtype=int)
batch_starts = np.random.choice(idx, size=n_iters + 1, replace=(n_iters + 1 > len(idx)))
tic = time.time()
loss_history = []


y_train_batch = get_batch(y_train, batch_starts[0], batch_size)
u_train_batch = get_batch(u_train, batch_starts[0], batch_size)

fig2 = plt.figure(2)
ax1 = fig2.add_subplot(311)
line1, = ax1.plot(t_sim, u_train_batch)
line2, = ax1.plot(t_sim, u_train_batch)
ax1.set_ylim((min(u_train), max(u_train)))
ax2 = fig2.add_subplot(312)
line3, = ax2.plot(t_sim, y_train_batch[:,0])
line4, = ax2.plot(t_sim, [0]*batch_size)
ax2.set_ylim((np.min(y_train[:,0]), np.max(y_train[:,0])))
ax3 = fig2.add_subplot(313)
line5, = ax3.plot(t_sim, y_train_batch[:,1])
line6, = ax3.plot(t_sim, [0]*batch_size)
ax3.set_ylim((np.min(y_train[:,1]), np.max(y_train[:,1])))

u0_batch = tf.cast(u_train_batch[0], dtype=tf.float32)
y0_batch = y_train_batch[0]
init_state_batch = tf.concat([u0_batch, y0_batch], axis=0)

# final_y, y_history = neural_ode.forward(init_state_batch, u_train_batch, return_states="tf") used for test

for step in range(n_iters + 1):
    y_train_batch = get_batch(y_train, batch_starts[step], batch_size)
    u_train_batch = get_batch(u_train, batch_starts[step], batch_size)
    u0_batch = tf.cast(u_train_batch[0], dtype=tf.float32)
    y0_batch = y_train_batch[0]
    init_state_batch = tf.concat([u0_batch, y0_batch, y0_p], axis=0)
    loss, x_history = compute_gradients_and_update_path(init_state_batch, u_train_batch, y_train_batch)
    loss_history.append(loss.numpy())

    line1.set_ydata(u_train_batch)
    line2.set_ydata(x_history[:, 0])
    line3.set_ydata(y_train_batch[:, 0])
    line4.set_ydata(x_history[:, 1])
    line5.set_ydata(y_train_batch[:, 1])
    line6.set_ydata(x_history[:, 2])
    fig2.canvas.draw()
    fig2.canvas.flush_events()

    if step % int(n_iters/100) == 0:
        percentage = int(100*step/n_iters)
        s1 = '{0:.0f}%'.format(percentage)
        s2 = 'Iteration: {0:.0f}'.format(step) + ' Of {0:.0f}'.format(n_iters)
        s3 = 'loss: {0:.4f}'.format(loss.numpy())
        print(s1, s2, s3, sep=', ')

toc = time.time()
training_time = toc - tic
print("Training time: ", training_time)

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


plt.figure(4)
plt.title("Loss per iteration")
plt.plot(range(n_iters + 1), loss_history)


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

model.save_weights("CSTR_SecondOrder_MLP_v4_1.ckpt")

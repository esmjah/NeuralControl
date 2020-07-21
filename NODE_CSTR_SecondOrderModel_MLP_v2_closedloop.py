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


def descale_data(scaled_1d_array, y_offset, y_scale):
    size_ = len(scaled_1d_array)
    original_data = np.zeros(size_)
    for j in range(size_):
        original_data[j] = y_offset[j] + scaled_1d_array[j]*(y_scale[j])
    return original_data


def scale_data(original_1d_array, y_offset, y_scale):
    size_ = len(original_1d_array)
    scaled_data = np.zeros(size_)
    for j in range(size_):
        scaled_data[j] = (original_1d_array[j] - y_offset[j])/y_scale[j]
    return scaled_data


data = io.loadmat(file_name='cstr_data_1.mat')

dt = data['dt'][0][0]
z0 = data['z0'][0]
t_process = data['t_process'][0]
u_process = data['u_process']
y_process = data['y_process']


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
        self.dense3 = keras.layers.Dense(32, activation='tanh', name='dense_3')
        self.dense4 = keras.layers.Dense(2, activation='linear', name='dense_4')
        self.dense5 = keras.layers.Dense(32, input_shape=(1, 5), activation='linear', name='dense_5')
        self.dense6 = keras.layers.Dense(32, activation='tanh', name='dense_6')
        self.dense7 = keras.layers.Dense(32, activation='tanh', name='dense_7')
        self.dense8 = keras.layers.Dense(4, activation='linear', name='dense_8')
        self.Kp1 = tf.Variable(tf.cast(1.0, dtype=tf.float32))
        self.Kp2 = tf.Variable(tf.cast(-1.0, dtype=tf.float32))
        self.Tau1 = tf.Variable(tf.cast(1.0, dtype=tf.float32))
        self.Tau2 = tf.Variable(tf.cast(1.0, dtype=tf.float32))
        self.z1 = tf.Variable(tf.cast(0.5, dtype=tf.float32))
        self.z2 = tf.Variable(tf.cast(0.5, dtype=tf.float32))

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
        x1_p_dot = ((self.Kp1/self.Tau1)*u6[0] - 2.0*(self.z1*x1) - x1_p/self.Tau1)/self.Tau1
        x2_dot = x2_p
        x2_p_dot = ((self.Kp2/self.Tau2)*u6[1] - 2.0*(self.z2*x2) - x2_p/self.Tau2)/self.Tau2
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
model.load_weights("CSTR_SecondOrder_MLP_v2_1.ckpt")


#  ************* Controller starts here ****************

Ts = 0.1  # Sampling interval of the simulation loop
t_iter = np.linspace(0, Ts, num=11)  # Divide Ts to 10 smaller steps for better accuracy of ODE solver
# Scaling x0 to original process
x0 = y_process[0]
x_k = x0
neural_ode_test = NeuralODE(model, t=t_iter)

T_sim = 10  # final simulation time
N_sim = int(T_sim / Ts)  # Number of iterations of the simulation loop
x_process = np.array([x0])
t_process = np.linspace(0, T_sim, N_sim + 1)  # Simulate the process for 4 minutes.

# Controller variables
u0 = u_process[0]
# u0 = x0[2]
u_k_1 = u0
u_control = np.array([u0])
IntegralError = 0.0
Error_k_1 = IntegralError

# Controller parameters
Kc0 = 0.2  # 0.2 ok
Kc = scale_u*Kc0/scale_y[0]
Ti = 0.1  # 0.1 ok
Td = 0.05  # 0.05 ok
setpoint_iter = x0[0]
setpoint = np.array([setpoint_iter])
uMin = u0 - scale_u
uMax = u0 + scale_u
du_max = scale_u * 0.1

uk_PID_fig = u0
y0_p = tf.cast([0, 0], dtype=tf.float32)

states_final = init_state

for k in tqdm(range(N_sim)):
    if k > N_sim/2:
        setpoint_iter = 0.09
    setpoint = np.append(setpoint, setpoint_iter)
    Error_k = np.array([setpoint_iter - x_k[0]])
    IntegralError += Error_k * Ts
    d_Error_dt = (Error_k - Error_k_1) / Ts
    uk_PID = u0 + Kc * (Error_k + IntegralError / Ti + Td * d_Error_dt)
    uk_PID_fig = np.append(uk_PID_fig, uk_PID, axis=0)
    du = uk_PID[0] - u_k_1

    if np.abs(du) <= du_max:
        u_k = uk_PID[0]
    else:
        u_k = u_k_1 + np.sign(du) * du_max

    u_k = np.clip(u_k, uMin, uMax)

    u_k_1 = u_k
    Error_k_1 = Error_k

    # Scale to the NN dimensions
    x_k_ = tf.cast(scale_data(x_k, y_process[0], scale_y), dtype=tf.float32)
    u_k_ = scale_data(np.array([u_k]), u_process[0], scale_u)
    u_iter = np.repeat(u_k_, len(t_iter))
    u_iter.shape = (len(t_iter), 1)

    init_state_iter = tf.concat([u_k_, states_final[1:len(states_final)]], axis=0)
    states_final, predicted_y_test = neural_ode_test.forward(init_state_iter, u_iter, return_states="numpy")

    x_temp = states_final[1:len(x0) + 1]
    # descale the NN output to original process dimensions
    x_k = descale_data(x_temp, y_process[0], scale_y)
    x_process = np.append(x_process, np.array([x_k]), axis=0)


x_1 = x_process[:, 0]
x_2 = x_process[:, 1]
u = uk_PID_fig

plt.close('all')
plt.figure(5)
plt.subplot(411)
plt.title("Network simulation with PI controller")
plt.plot(t_process, x_1, label="x_1")
plt.plot(t_process, setpoint, label="setpoint")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
#plt.legend()
plt.subplot(412)
plt.plot(t_process, x_2,  label="x_2")
plt.savefig('output.png', dpi=600, format='png')
plt.legend()
plt.subplot(413)
plt.plot(t_process, u, label="u")
plt.legend()
plt.subplot(414)
plt.plot(t_process, uk_PID_fig, label="uk_PID")
plt.legend()

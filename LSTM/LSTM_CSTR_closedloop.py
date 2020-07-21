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


class Process:
    def __init__(self):
        super(Process, self).__init__()
        self.Fih=1.0
        self.q = 100.0
        self.Caf = 1.0
        self.Tf = 350.0
        self.Tcf = 350.0
        self.V = 100.0
        self.ha = 7.0*10**5
        self.k0 = 7.2*10**10
        self.ER=9.95*10**3
        self.deltaH=-2*10**5
        self.ER=9.95*10**3
        self.Ro=1000.0
        self.Roc=1000.0
        self.Cp=1.0
        self.Cpc=1.0
        self.Fic=1.0
        self.alfah=0.01

    def ode_model(self, t, z, u):
        x = z[0]
        y = z[1]
        dxdt=(self.q/self.V)*(self.Caf-x)-self.k0*x*np.exp(-self.ER/y)*self.Fic
#        dxdt = (-x + u)/self.Kp
        hd= self.ha # (1-self.alfah*t)*self.ha
        dydt=(self.q/self.V)*(self.Tf-y)+(-self.deltaH*self.k0*x/(self.Ro*self.Cp))*np.exp(-self.ER/y)*self.Fic+(self.Roc*self.Cpc/(self.Ro*self.Cp*self.V))*u*(1-np.exp(-hd/(u*self.Ro*self.Cpc)*self.Fih))*(self.Tcf-y)
#        dydt = (-y + x)/self.Tau
        dzdt = [dxdt,dydt]
        return dzdt


def simulate_process(f, t, u, x0):
    r = ode(f).set_integrator('zvode', method='bdf')
    xk = x0
    t_values = [t[0]]
    x_values = [xk]

    deltas_t = t[1:] - t[:-1]
    for it in range(len(deltas_t)):
        r.set_initial_value(xk, t[it]).set_f_params(u[it])
        xk = np.real(r.integrate(r.t + deltas_t[it])).tolist()
        if r.successful():
            t_values.append(r.t)
            x_values.append(xk)

    return np.array(t_values), np.array(x_values)


def descale_2d_data(scaled_2d_array, y_offset, y_scale):
    len_ = scaled_2d_array.shape[0]
    col_ = scaled_2d_array.shape[1]
    original_data = np.zeros((len_, col_))
    for j in range(len_):
        original_data[j, :] = y_offset + scaled_2d_array[j, :]*y_scale
    return original_data


def scale_2d_data(original_2d_array, y_offset, y_scale):
    len_ = original_2d_array.shape[0]
    col_ = original_2d_array.shape[1]
    scaled_data = np.zeros((len_, col_))
    for j in range(len_):
        scaled_data[j,:] = (original_2d_array[j, :] - y_offset)/y_scale
    return scaled_data


def descale_1d_data(scaled_1d_array, y_offset, y_scale):
    size_ = len(scaled_1d_array)
    original_data = np.zeros(size_)
    for j in range(size_):
        original_data[j] = y_offset[j] + scaled_1d_array[j]*(y_scale[j])
    return original_data


def scale_1d_data(original_1d_array, y_offset, y_scale):
    size_ = len(original_1d_array)
    scaled_data = np.zeros(size_)
    for j in range(size_):
        scaled_data[j] = (original_1d_array[j] - y_offset[j])/y_scale[j]
    return scaled_data


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
look_back = 25
no_of_features=3  # Ca, T ,u

trainX, trainY = create_dataset(train_data, look_back)
n_train = trainX.shape[0]


u0 = tf.cast(u_train[0], dtype=tf.float32)
y0 = y_train[0]
y0_p = tf.cast([0, 0], dtype=tf.float32)

init_state = tf.concat([u0, y0, y0_p], axis=0)

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

p = Process()

#  ************* Controller starts here ****************

Ts = 0.1  # Sampling interval of the simulation loop
t_iter = np.linspace(0, Ts, num=2)
# Scaling x0 to original process
y0 = y_process[0]
y_k = y0

T_sim = 10  # final simulation time
N_sim = int(T_sim / Ts)  # Number of iterations of the simulation loop
y_lstm = np.array([y0])
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
setpoint_iter = y0[0]
setpoint_lstm = np.array([setpoint_iter])
uMin = u0 - scale_u
uMax = u0 + scale_u
du_max = scale_u * 0.1

uk_PID_lstm = u0
y0_p = tf.cast([0, 0], dtype=tf.float32)

states_final = init_state

for k in tqdm(range(N_sim)):
    if k > N_sim/4:
        setpoint_iter = 0.09
    setpoint_lstm = np.append(setpoint_lstm, setpoint_iter)
    Error_k = np.array([setpoint_iter - y_k[0]])
    IntegralError += Error_k * Ts
    d_Error_dt = (Error_k - Error_k_1) / Ts
    uk_PID = u0 + Kc * (Error_k + IntegralError / Ti + Td * d_Error_dt)
    uk_PID_lstm = np.append(uk_PID_lstm, uk_PID, axis=0)
    du = uk_PID[0] - u_k_1

    if np.abs(du) <= du_max:
        u_k = uk_PID[0]
    else:
        u_k = u_k_1 + np.sign(du) * du_max

    u_k = np.clip(u_k, uMin, uMax)

    u_k_1 = u_k
    Error_k_1 = Error_k

    if k > look_back:
        # Scale to the NN dimensions
        y_k_ = np.array(y_lstm[k - look_back - 1:k])
        u_k_ = np.array(uk_PID_lstm[k - look_back - 1:k])
        y_k_scaled = scale_2d_data(np.reshape(y_k_, (look_back+1, 2)), y_process[0], scale_y)
        u_k_scaled = scale_2d_data(np.reshape(u_k_, (look_back+1, 1)), u_process[0], scale_u)

        lstm_data = np.concatenate((y_k_scaled, u_k_scaled), axis=1)
        testX, testY = create_dataset(lstm_data, look_back)

        y_predicted = model.predict(testX)

        # descale the NN output to original process dimensions
        y_k = descale_1d_data(y_predicted[0, [0, 1]], y_process[0], scale_y)
    y_lstm = np.append(y_lstm, np.array([y_k]), axis=0)


x_1_lstm = y_lstm[:, 0]
x_2_lstm = y_lstm[:, 1]
u_PID_lstm = uk_PID_lstm

# ******************* Simulation of real process with PID **********************

x_k = z0
u_PID_process = u0
setpoint_iter = x_k[0]
x_process = np.array([x_k])
setpoint_process = np.array([setpoint_iter])

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

for k in tqdm(range(N_sim)):
    if k > N_sim/4:
        setpoint_iter = 0.09
    setpoint_process = np.append(setpoint_process, setpoint_iter)
    Error_k = np.array([setpoint_iter - x_k[0]])
    IntegralError += Error_k * Ts
    d_Error_dt = (Error_k - Error_k_1) / Ts
    uk_PID = u0 + Kc * (Error_k + IntegralError / Ti + Td * d_Error_dt)
    u_PID_process = np.append(u_PID_process, uk_PID, axis=0)
    du = uk_PID[0] - u_k_1

    if np.abs(du) <= du_max:
        u_k = uk_PID[0]
    else:
        u_k = u_k_1 + np.sign(du) * du_max

    u_k = np.clip(u_k, uMin, uMax)

    u_k_1 = u_k
    Error_k_1 = Error_k

    x0_process = x_k
    u_iter = np.repeat(u_k, len(t_iter))
    u_iter.shape = (len(t_iter), 1)
    t_, x_temp = simulate_process(f=p.ode_model, t=t_iter, u=u_iter, x0=x0_process)
    x_k = x_temp[-1]
    x_process = np.append(x_process, np.array([x_k]), axis=0)


x_1_process = x_process[:, 0]
x_2_process = x_process[:, 1]

plt.close('all')
plt.figure(1)
plt.subplot(311)
plt.title("Network simulation with PI controller")
plt.plot(t_process, x_1_process, label="x_1 process")
plt.plot(t_process, x_1_lstm, linestyle='dashed',  label="x_1 LSTM")
plt.plot(t_process, setpoint_process, label="setpoint")
plt.legend()
plt.subplot(312)
plt.plot(t_process, x_2_process,  label="x_2 process")
plt.plot(t_process, x_2_lstm, linestyle='dashed', label="x_2 LSTM")
plt.legend()
plt.subplot(313)
plt.plot(t_process, u_PID_process, label="u process")
plt.plot(t_process, u_PID_lstm, linestyle='dashed', label="u LSTM")
plt.legend()
plt.savefig('output.png', dpi=600, format='png')

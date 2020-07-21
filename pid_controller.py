#%%
import numpy as np
import numpy.random as npr
from sklearn import preprocessing
import tensorflow as tf
tf.executing_eagerly()
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import pandas
from scipy.integrate import ode
keras = tf.keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
#import control as c
from neural_ode_u import NeuralODE
import h5py
###################### PID Controller ################################
#model.save_weights('my_model_weights.h5')
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
def pseudo_random_pulse(num_value, period, average_value, min_val, max_val):
    sig = np.zeros((num_value, 1))
    deviation = abs(max_val - min_val)
    j = 0
    new_val = average_value
    for k in range(num_value):
        sig[k] = new_val
        j = j + 1
        if j == period:
            new_val = np.clip(average_value + deviation*(np.random.rand()-0.5), min_val, max_val)
            j = 0
    return sig

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

    return t_values, x_values
def data_normalizer(input_data):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    np_scaled = min_max_scaler.fit_transform(input_data)
    data_max = min_max_scaler.data_max_
    data_min = min_max_scaler.data_min_
    input_data_normalized = pandas.DataFrame(np_scaled)
    input_data_n=input_data_normalized
    input_data_n=np.array(input_data_n.loc[:,:])
    return input_data_n, data_min, data_max


def descale_data(scaled_1d_array, min_data, max_data):
    size_ = len(scaled_1d_array)
    original_data = np.zeros(size_)
    for j in range(size_):
        original_data[j] = min_data[j] + scaled_1d_array[j]*(max_data[j] - min_data[j])
    return original_data


def scale_data(original_1d_array, min_data, max_data):
    size_ = len(original_1d_array)
    scaled_data = np.zeros(size_)
    for j in range(size_):
        scaled_data[j] = (original_1d_array[j] - min_data[j])/(max_data[j] - min_data[j])
    return scaled_data

class SOPTD(tf.keras.Model):
    def __init__(self):
        super(SOPTD, self).__init__()
        self.dense1 = keras.layers.Dense(16, input_shape=(1, 3), activation='linear', name='dense_1')
        self.dense2 = keras.layers.Dense(16, activation='tanh', name='dense_2')
        self.dense3 = keras.layers.Dense(8, activation='tanh', name='dense_3')
        self.dense4 = keras.layers.Dense(2, activation='linear', name='dense_4')
        
        self.dense5 = keras.layers.Dense(24, input_shape=(1, 4), activation='linear', name='dense_5')
        self.dense6 = keras.layers.Dense(16, activation='tanh', name='dense_6')
        self.dense7 = keras.layers.Dense(8, activation='tanh', name='dense_7')
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
        x_dot_ = tf.concat([[x1_dot], [x2_dot], [x1_p_dot], [x2_p_dot]], axis=0)
        x_dot = tf.reshape(x_dot_, [1, 4])
        g1 = self.dense5(x_dot)
        g2 = self.dense6(g1)
        g3 = self.dense7(g2)
        g4 = self.dense8(g3)
        g5 = tf.reshape(g4, [4, ])

        u_dot = tf.reshape(tf.cast(0, dtype=tf.float32), [1, ])
        h = tf.concat([u_dot, g5], axis=0)
        return h
model = SOPTD()    
p = Process()
z0=[0.07458431103457913, 442.58378163693106]

t_final = 100
dt = 0.1
num_test_data = int(t_final/dt)
t_test = np.linspace(0, t_final-dt, num=num_test_data)
u_test = pseudo_random_pulse(num_test_data, int(num_test_data/50), 100, 90, 110)
# T, y_test, x_test = c.forced_response(G, T=t_test, U=u_test, X0=1)
t_, y_test = simulate_process(f=p.ode_model, t=t_test, u=u_test, x0=z0)

total_data_orig=np.concatenate((np.array(y_test),u_test),axis=1)

total_data1, min_orig, max_orig =data_normalizer(total_data_orig)


Ts = 0.01  # Sampling interval of the simulation loop
t_iter = np.linspace(0, Ts, num=11)  # Divide Ts to 10 smaller steps for better accuracy of ODE solver
# Scaling x0 to original process
min_scale = min_orig[1:4]
max_scale = max_orig[1:4]
x0 = z0
x_k = x0
neural_ode_test = NeuralODE(model, t= t_iter)

T_sim = 10  # final simulation time
N_sim = int(T_sim/Ts)  # Number of iterations of the simulation loop
x_process = np.array([x0])
t_process = np.linspace(0, T_sim, N_sim + 1)  # Simulate the process for 4 minutes.

# Controller variables
u0 =50.0 
#u0 = x0[2]
u_k_1 = u0
u_control = np.array([u0])
IntegralError = 0.0
Error_k_1 = IntegralError

scale_x = max_scale - min_scale
scale_y = scale_x[0]
scale_u = scale_x[2]
# Controller parameters
Kc0 = 0.2   # 0.2 ok
Kc = scale_u*Kc0/scale_y
Ti = 0.1  #  0.1 ok 
Td = 0.05 #  0.05 ok 
setpoint = 0.085
du_max = scale_u*0.1
uMin = min_scale[2]
uMax = max_scale[2]
uk_PID_fig=np.array([u0])
y0_p = tf.cast([0, 0], dtype=tf.float32)
for k in tqdm(range(N_sim)):
    Error_k = np.array([setpoint - x_k[0]])
    IntegralError += Error_k * Ts
    d_Error_dt = (Error_k - Error_k_1) / Ts
    uk_PID = u0 + Kc * (Error_k + IntegralError / Ti + Td * d_Error_dt)
    uk_PID_fig= np.append(uk_PID_fig, uk_PID, axis=0)
    du = uk_PID[0] - u_k_1

    if np.abs(du) <= du_max:
        u_k = uk_PID[0]
    else:
        u_k = u_k_1 + np.sign(du) * du_max

    u_k = np.clip(u_k, uMin, uMax)

    u_k_1 = u_k
    Error_k_1 = Error_k

    # Scale to the NN dimensions
    x_k_ = scale_data(np.array([x_k[0], x_k[1], u_k]), min_scale, max_scale)
    u_k_=scale_data(np.array(u_k), min_scale, max_scale)
    init_state_test=tf.concat([u_k_, x_k_, y0_p], axis=0)
    u_test_scaled=u_k_
    
    states_temp, predicted_y_test = neural_ode_test.forward(init_state_test, u_test_scaled, return_states="numpy")
    
    x_temp = np.concatenate(states_temp)
    # descale the NN output to original process dimensions
    x_k = descale_data(x_temp[-1], min_scale, max_scale)
    x_process = np.append(x_process, np.array([x_k]), axis=0)

x_1 = x_process[:, 0]
x_2 = x_process[:, 1]
u = x_process[:, 2]

plt.figure(5)
plt.subplot(411)
plt.title("Network simulation with PI controller")
plt.plot(t_process, x_1, label="x_1")
plt.plot(t_process, setpoint*np.ones(len(t_process)), label="setpoint")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
#plt.legend()
plt.subplot(412)
plt.plot(t_process, x_2,  label="x_2")
plt.savefig('output.jpg', dpi=600, format='jpg')
plt.legend()
plt.subplot(413)
plt.plot(t_process, u, label="u")
plt.legend()
plt.subplot(414)
plt.plot(t_process, uk_PID_fig, label="uk_PID")
plt.legend()

#plt.show()

plt.figure(6)
fig = plt.figure()
plt.plot(t_process, u-uk_PID_fig)
fig.savefig('diffr.jpg', dpi=300)
plt.show()
print(u[-1]-uk_PID_fig[-1])
#%ppppppppppppppppppppppppppppppp
model.summary()

#savefig('output.jpg', dpi=600, facecolor='w', edgecolor='w',
#        orientation='portrait', papertype=None, format=None,
#        transparent=False, bbox_inches=None, pad_inches=0.1,
#        frameon=None, metadata=None)
#plt.savefig('output.jpg', dpi=600, format='jpg')
#import matplotlib.pyplot.savefig
#savefig(fname, dpi=None, facecolor='w', edgecolor='w',
#        orientation='portrait', papertype=None, format=None,
#        transparent=False, bbox_inches=None, pad_inches=0.1,
#        frameon=None, metadata=None)
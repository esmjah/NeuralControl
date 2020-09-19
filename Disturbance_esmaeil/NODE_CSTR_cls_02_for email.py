import numpy as np
import numpy.random as npr
import tensorflow as tf
import tensorflow.contrib.eager as tfe
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

#tf.executing_eagerly()
tf.enable_eager_execution()

#============================ simulate process for training data ====================

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
    return np.array(sig)


#######################################################################
class Process:
    def __init__(self):
        super(Process, self).__init__()
        self.Fih=1.0
#        self.q = 100.0
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

    def ode_model(self, t, z, u1,q1):
        x = z[0]
        y = z[1]
        u=u1[0]
        q=q1[0]
        dxdt=(q/self.V)*(self.Caf-x)-self.k0*x*np.exp(-self.ER/y)*self.Fic
#        dxdt = (-x + u)/self.Kp
        hd= self.ha # (1-self.alfah*t)*self.ha
        dydt=(q/self.V)*(self.Tf-y)+(-self.deltaH*self.k0*x/(self.Ro*self.Cp))*np.exp(-self.ER/y)*self.Fic+(self.Roc*self.Cpc/(self.Ro*self.Cp*self.V))*u*(1-np.exp(-hd/(u*self.Ro*self.Cpc)*self.Fih))*(self.Tcf-y)
#        dydt = (-y + x)/self.Tau
        dzdt = [dxdt,dydt]
        return dzdt

#X = np.column_stack((u_process, u_process))

def simulate_process(f, t, u, q, x0):
    r = ode(f).set_integrator('zvode', method='bdf')
    xk = x0
    t_values = [t[0]]
    x_values = [xk]

    deltas_t = t[1:] - t[:-1]
    for it in range(len(deltas_t)):
        r.set_initial_value(xk, t[it]).set_f_params(u[it], q[it])
        xk = np.real(r.integrate(r.t + deltas_t[it])).tolist()
        if r.successful():
            t_values.append(r.t)
            x_values.append(xk)

    return np.array(t_values), np.array(x_values)


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

#==============================================
dt = 0.1
t_final = 310
num_train_data = int(t_final/dt)
t_process = np.linspace(0, t_final-dt, num=num_train_data)

u1_process = pseudo_random_pulse(num_train_data, 30, 100, 90, 110)
q_process = pseudo_random_pulse(num_train_data, 30, 100, 95, 105)

p = Process()
# z0=[0.07,441.0]
z0=[0.07458431103457913, 442.58378163693106]
t_, y_process = simulate_process(f=p.ode_model, t=t_process, u=u1_process, q=q_process, x0=z0)
z0=np.array([0.07458431103457913, 442.58378163693106])
t_process = t_
u_process = np.column_stack((u1_process, q_process))
y_process = y_process


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
        self.dense1 = keras.layers.Dense(4, input_shape=(1, 4), activation='linear', name='dense_1')
        self.dense2 = keras.layers.Dense(128, activation='tanh', name='dense_2')
        self.dense3 = keras.layers.Dense(128, activation='relu', name='dense_3')
        self.dense4 = keras.layers.Dense(2, activation='linear', name='dense_4')
        self.dense5 = keras.layers.Dense(128, input_shape=(1, 5), activation='linear', name='dense_5')
        self.dense6 = keras.layers.Dense(128, activation='tanh', name='dense_6')
        self.dense7 = keras.layers.Dense(64, activation='relu', name='dense_7')
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
        q = states[1]
        x1 = states[2]
        x2 = states[3]
        x1_p = states[4]
        x2_p = states[5]
        u1 = tf.reshape([u, q, x1, x2], [1, 4])
        u2 = self.dense1(u1)
        u3 = self.dense2(u2)
        u4 = self.dense3(u3)
        u5 = self.dense4(u4)
        u6 = tf.reshape(u5, [2, ])
#        u =    u6[0]
#        x1 =   u6[1]
#        x2 =   u6[2]
#        x1_p = u6[3]
#        x2_p = u6[4]
        x1_dot = x1_p
        x1_p_dot = ((self.Kp1/self.Tau1)*u6[0] - 2.0*(self.z1*x1_p) - x1/self.Tau1)/self.Tau1
        x2_dot = x2_p
        x2_p_dot = ((self.Kp2/self.Tau2)*u6[1] - 2.0*(self.z2*x2_p) - x2/self.Tau2)/self.Tau2
        x_dot = tf.concat([[x1_dot], [x2_dot], [x1_p_dot], [x2_p_dot], [u]], axis=0)
        g0 = tf.reshape(x_dot, [1, 5])
        g1 = self.dense5(g0)
        g2 = self.dense6(g1)
        g3 = self.dense7(g2)
        g4 = self.dense8(g3)
        g5 = tf.reshape(g4, [4, ])

        u_dot = tf.zeros(2)
        h = tf.concat([u_dot, g5], axis=0)
        return h



model = SOPTD()
#model.load_weights("CSTR_SecondOrder_MLP_v3_1.ckpt")
#model.load_weights("CSTR_SecondOrder_MLP_v4_7my.ckpt")
#model.load_weights("CSTR_SecondOrder_MLP_one_dist03my_OK.ckpt")
model.load_weights("CSTR_SecondOrder_MLP_one_dist03my.ckpt")

#model.load_weights("CSTR_SecondOrder_MLP_v4_7resmy.ckpt")

p = Process()

#  ************* Controller starts here ****************

Ts = 0.1  # Sampling interval of the simulation loop
t_iter = np.linspace(0, Ts, num=11)  # Divide Ts to 10 smaller steps for better accuracy of ODE solver
## Scaling x0 to original process
x0 = y_process[0]
x_k = x0
neural_ode_test = NeuralODE(model, t=t_iter)

T_sim = 45  # final simulation time
N_sim = int(T_sim / Ts)  # Number of iterations of the simulation loop
x_node = np.array([x0])
t_process = np.linspace(0, T_sim, N_sim + 1)  # Simulate the process for 4 minutes.

## Controller variables
u0 =np.array([u_process[0][0]])

u_k_1 = u0
u_control = np.array([u0])
IntegralError = 0.0
Error_k_1 = IntegralError

# Controller parameters
Kc0 = 0.2  # 0.2 ok
Kc = scale_u[0]*Kc0/scale_y[0]
Ti = 0.1  # 0.1 ok
Td = 0.05  # 0.05 ok
setpoint_iter = x0[0]
distur_iter   = u_process[0][1]  # q in model has been chosen as disturbance
setpoint_node = np.array([setpoint_iter])
distur_node = np.array([distur_iter])
uMin = u0 - scale_u
uMin = uMin[0]
uMax = u0 + scale_u
uMax = uMax[0]
du_max = scale_u * 0.1
du_max = du_max [0]

uk_PID_fig = u0
y0_p = tf.cast([0, 0], dtype=tf.float32)
#
states_final = init_state
#
for k in tqdm(range(N_sim)):
    if k > N_sim/9:
        setpoint_iter = 0.09
        distur_iter   = 100
    if k > N_sim/3:
        setpoint_iter = 0.09
        distur_iter   = 105
    if k > N_sim/1.5:
        setpoint_iter = 0.09    
        distur_iter   = 100
        
#        if k > N_sim/4:
#        setpoint_iter = 0.09
    setpoint_node = np.append(setpoint_node, setpoint_iter)
    Error_k = np.array([setpoint_iter - x_k[0]])
    IntegralError += Error_k * Ts
    d_Error_dt = (Error_k - Error_k_1) / Ts
    uk_PID = u0 + Kc * (Error_k + IntegralError / Ti + Td * d_Error_dt)
    
    du = uk_PID[0] - u_k_1

    if np.abs(du) <= du_max:
        u_k = uk_PID[0]
    else:
        u_k = u_k_1 + np.sign(du) * du_max

    u_k = np.clip(u_k, uMin, uMax)
    
    uk_PID = np.array([u_k])
    uk_PID_fig = np.append(uk_PID_fig, uk_PID, axis=0)

    u_k_1 = u_k
    Error_k_1 = Error_k

    # Scale to the NN dimensions
    x_k_ = tf.cast(scale_data(x_k, y_process[0], scale_y), dtype=tf.float32)
    u_k_ = scale_data(np.array([u_k]), np.array([u_process[0][0]]), np.array([scale_u[0]]))
    distur_iter_ = scale_data(np.array([distur_iter]), np.array([u_process[0][1]]), np.array([scale_u[1]]))
    
    u_iter = np.repeat(u_k_, len(t_iter))
    u_iter.shape = (len(t_iter), 1)
    
    
    q_iter = np.repeat(distur_iter_, len(t_iter))
    q_iter.shape = (len(t_iter), 1)

    init_state_iter_ = np.concatenate((u_k_, distur_iter_), axis=0)      
    init_state_iter = tf.concat([init_state_iter_, states_final[2:len(states_final)]], axis=0)
#    init_state_iter = tf.concat([u_k_, states_final[1:len(states_final)]], axis=0)
#    print('u_iter = ',u_iter)
    unode = np.concatenate((u_iter, q_iter), axis=1)
       
    
    states_final, predicted_y_test = neural_ode_test.forward(init_state_iter, unode, return_states="numpy")
##
    x_temp = states_final[2:len(x0) + 2]
    # descale the NN output to original process dimensions
    x_k = descale_data(x_temp, y_process[0], scale_y)
    x_node = np.append(x_node, np.array([x_k]), axis=0)
#    distur_iter_ = distur_iter
#
#
x_1_node = x_node[:, 0]
x_2_node = x_node[:, 1]
u_PID_node = uk_PID_fig
#
## ******************* Simulation of real process with PID **********************
#
x_k = z0
u_PID_process = u0
#u_PID_process = np.array(u0)
setpoint_iter = x_k[0]
x_process = np.array([x_k])
setpoint_process = np.array([setpoint_iter])
q1_iter = 100
#
## Controller variables
#u0 = u_process[0][0]
u0 = np.array([u_process[0][0]])
# u0 = x0[2]
u_k_1 = u0
u_control = np.array([u0])
IntegralError = 0.0
Error_k_1 = IntegralError

# Controller parameters
Kc0 = 0.2  # 0.2 ok
Kc = scale_u[0]*Kc0/scale_y[0]
Ti = 0.1  # 0.1 ok
Td = 0.05  # 0.05 ok


for k in tqdm(range(N_sim)):
    if k > N_sim/9:
        setpoint_iter = 0.09
        q1_iter = 100
    if k > N_sim/3:
        setpoint_iter = 0.09
        q1_iter = 105
    if k > N_sim/1.5:
        setpoint_iter = 0.09 
        q1_iter = 100
#    if k > N_sim/4:
#        setpoint_iter = 0.09
    setpoint_process = np.append(setpoint_process, setpoint_iter)
    Error_k = np.array([setpoint_iter - x_k[0]])
    IntegralError += Error_k * Ts
    d_Error_dt = (Error_k - Error_k_1) / Ts
    uk_PID = u0 + Kc * (Error_k + IntegralError / Ti + Td * d_Error_dt)
#    print('uk_PID=',uk_PID)
#    input('enter')
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
    
    q1_iter1 = np.array(q1_iter)
    q1_iter1 = np.repeat(q1_iter, len(t_iter))
    q1_iter1.shape = (len(t_iter), 1)
    
    
    t_, x_temp = simulate_process(f=p.ode_model, t=t_iter, u=u_iter, q=q1_iter1, x0=x0_process)
#    t_, x_temp = simulate_process(f=p.ode_model, t=t_process, u=u1_process, q=q_process, x0=z0)**********
    x_k = x_temp[-1]
    x_process = np.append(x_process, np.array([x_k]), axis=0)
#    input('sdsd')
#    print(q1_iter)


x_1_process = x_process[:, 0]
x_2_process = x_process[:, 1]

plt.close('all')
plt.figure(1)
plt.subplot(311)
plt.title("Network simulation with PI controller")
plt.plot(t_process, x_1_process, label="x_1 process")
plt.plot(t_process, x_1_node, linestyle='dashed',  label="x_1 NODE")
plt.plot(t_process, setpoint_process, label="setpoint")
plt.legend()
plt.subplot(312)
plt.plot(t_process, x_2_process,  label="x_2 process")
plt.plot(t_process, x_2_node, linestyle='dashed', label="x_2 NODE")
plt.legend()
plt.subplot(313)
plt.plot(t_process, u_PID_process, label="u process")
plt.plot(t_process, u_PID_node, linestyle='dashed', label="u NODE")
plt.legend()
#plt.savefig('output.png', dpi=600, format='png')
##-------------------------------------------------------------------
#
#model_error = np.sum(np.square(x_1_process - x_1_node), axis=0)
#print("model_error_x1: ", model_error)
#
#model_error = np.sum(np.square(x_2_process - x_2_node), axis=0)
#print("model_error_x2: ", model_error)
#
#model_error = np.sum(np.square(u_PID_process - u_PID_node), axis=0)
#print("model_error_u_PID: ", model_error)
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
from neural_ode_u import NeuralODE
import h5py
from numpy.random import seed
from scipy import io
seed(0)
tf.random.set_seed(42)
tf.executing_eagerly()

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

y1_process = [y_[0] for y_ in y_process]
y2_process = [y_[1] for y_ in y_process]
#==============================================
plt.figure(3)
plt.plot(t_,u1_process,'g',label='u(t)')
plt.plot(t_,q_process,'r',label='q(t)')
plt.figure(4)
#plt.subplot(211)
plt.plot(t_,y1_process,'r',label='y1(t)')
plt.figure(5)
plt.plot(t_,y2_process,'b',label='y2(t)')



#========================================= end of process simulation for training data ============================================
def get_batch(x, start, batch_length):
    x_batch = x[int(start):int(start)+batch_length]
    return x_batch


#data = io.loadmat(file_name='cstr_data_1.mat')
#
#dt = data['dt'][0][0]
#z0 = data['z0'][0]
z0=np.array([0.07458431103457913, 442.58378163693106])
t_process = t_
u_process = np.column_stack((u1_process, q_process))
y_process = y_process


y1_process = [y_[0] for y_ in y_process]
y2_process = [y_[1] for y_ in y_process]
#
plt.close('all')

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
y0 = tf.cast(y_train[0], dtype=tf.float32)
y0_p = tf.cast([0, 0], dtype=tf.float32)

init_state = tf.concat([u0, y0, y0_p], axis=0)


class SOPTD(tf.keras.Model):
    def __init__(self):
        super(SOPTD, self).__init__()
        self.dense1 = keras.layers.Dense(4, input_shape=(1, 4), activation='linear', name='dense_1')
        self.dense2 = keras.layers.Dense(64, activation='tanh', name='dense_2')
        self.dense3 = keras.layers.Dense(64, activation='relu', name='dense_3')
        self.dense4 = keras.layers.Dense(2, activation='linear', name='dense_4')
        self.dense5 = keras.layers.Dense(64, input_shape=(1, 5), activation='linear', name='dense_5')
        self.dense6 = keras.layers.Dense(64, activation='tanh', name='dense_6')
        self.dense7 = keras.layers.Dense(32, activation='relu', name='dense_7')
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

# optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

model = SOPTD()

batch_size = 100
t_sim = np.linspace(0, (batch_size - 1)*dt, num=batch_size)

neural_ode = NeuralODE(model, t=t_sim)


def compute_gradients_and_update_path(initial_state, u_target, y_target):
    with tf.GradientTape() as g:
        final_y, x_history = neural_ode.forward(initial_state, u_target, return_states="tf")
        predicted_path = x_history[:, u_target.shape[1]:u_target.shape[1]+y_target.shape[1]]  # -> (batch_time, batch_size, 2)
        loss_ = tf.reduce_sum(tf.square(predicted_path - y_target), axis=0)
        prediction_loss = tf.reduce_sum(loss_)

    # back-propagate through solver with tensorflow
    gradients = g.gradient(prediction_loss, model.weights)
    optimizer.apply_gradients(zip(gradients, model.weights))
    return prediction_loss, x_history

# compute_gradients_and_update = tfe.defun(compute_gradients_and_update_path)
compute_gradients_and_update = tf.function(compute_gradients_and_update_path)

# ############## Training ################
n_iters = 1000
idx = np.arange(len(y_train) - batch_size, dtype=int)
batch_starts = np.random.choice(idx, size=n_iters + 1, replace=(n_iters + 1 > len(idx)))
tic = time.time()
loss_history = []


y_train_batch = get_batch(y_train, batch_starts[0], batch_size)
u_train_batch = get_batch(u_train, batch_starts[0], batch_size)


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

    if step % int(n_iters/100) == 0:
        percentage = int(100*step/n_iters)
        s1 = '{0:.0f}%'.format(percentage)
        s2 = 'Iteration: {0:.0f}'.format(step) + ' Of {0:.0f}'.format(n_iters)
        s3 = 'loss: {0:.4f}'.format(loss.numpy())
        print(s1, s2, s3, sep=', ')

toc = time.time()
training_time = toc - tic
print("Training time: ", training_time)
model.save_weights("CSTR_SecondOrder_MLP_dist.ckpt")
# ############## Test on training data ################
neural_ode_test = NeuralODE(model, t=t_train_data)
state_final, predicted_x = neural_ode_test.forward(init_state, u_train, return_states="numpy")
predicted_y = np.array(predicted_x[:,u_train.shape[1]:u_train.shape[1]+y_train.shape[1]])
loss_ = np.sum(np.square(predicted_y - y_train), axis=0)
training_loss = np.sum(loss_)
print("Training loss: ", training_loss)

plt.figure(6)
plt.suptitle("Training data")
plt.subplot(411)
plt.plot(t_train_data, u_train[:, 0])
plt.subplot(412)
plt.plot(t_train_data, u_train[:, 1])
plt.subplot(413)
plt.plot(t_train_data, y_train[:, 0])
plt.plot(t_train_data, np.array(predicted_y[:, 0]))
plt.subplot(414)
plt.plot(t_train_data, y_train[:, 1])
plt.plot(t_train_data, np.array(predicted_y[:, 1]))


plt.figure(4)
plt.title("Loss per iteration")
plt.plot(range(n_iters + 1), loss_history)


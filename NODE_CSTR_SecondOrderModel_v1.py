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

tf.executing_eagerly()


def get_batch(x, start, batch_length):
    x_batch = x[int(start):int(start)+batch_length]
    return x_batch


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


#######################################################################
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

    return t_values, x_values


#Kp = 2.0
#Tau = 10.0
#D = 15.0
#[numD,denD] = c.pade(D,3)
#G=c.tf( Kp , [ Tau , 1 ])

#print(G)

#A = -1/Tau
#B = Kp/Tau
#C = 1
#D = 0
#sys = c.StateSpace(A, B, C, D)

dt = 0.05
t_final = 300
num_train_data = int(t_final/dt)
t_process = np.linspace(0, t_final-dt, num=num_train_data)

u_process = pseudo_random_pulse(num_train_data, int(num_train_data/100), 100, 90, 110)

p = Process()
# z0=[0.07,441.0]
z0=[0.07458431103457913, 442.58378163693106]
t_, y_process = simulate_process(f=p.ode_model, t=t_process, u=u_process, x0=z0)

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

scaler_u = MinMaxScaler((-1, 1))
scaler_y = MinMaxScaler((-1, 1))

scaler_u.fit(u_process)
scaler_y.fit(y_process)
u_train = scaler_u.transform(u_process)
y_train = scaler_y.transform(y_process)
t_train_data = t_process
# max_scale_y = 0.1*np.max(y_process, axis=0)
# max_scale_u = 0.1*np.max(u_process)

#u_train = u_process/max_scale_u
#y_train = tf.cast(np.array(y_process)/max_scale_y, dtype=tf.float32)

u0 = tf.cast(u_train[0], dtype=tf.float32)
y0 = y_train[0]
y0_p = tf.cast([0, 0], dtype=tf.float32)

init_state = tf.concat([u0, y0, y0_p], axis=0)


class SOPTD(tf.keras.Model):
    def __init__(self):
        super(SOPTD, self).__init__()

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
        x1_dot = x1_p
        x1_p_dot = ((self.Kp1/self.Tau1)*u - 2.0*(self.z1*x1) - x1_p/self.Tau1)/self.Tau1
        x2_dot = x2_p
        x2_p_dot = ((self.Kp2/self.Tau2)*u - 2.0*(self.z2*x2) - x2_p/self.Tau2)/self.Tau2
        x_dot_ = tf.concat([[x1_dot], [x2_dot], [x1_p_dot], [x2_p_dot]], axis=0)
        x_dot = tf.reshape(x_dot_, [4, ])
        u_dot = tf.reshape(tf.cast(0, dtype=tf.float32), [1, ])
        h = tf.concat([u_dot, x_dot], axis=0)
        return h


# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)

model = SOPTD()
#for layer in model.layers: print(layer.get_config(), layer.get_weights())

batch_size = 200
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
n_iters = 3000
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
state_final, predicted_y = neural_ode_test.forward(init_state, u_train, return_states="numpy")


plt.figure(3)
plt.suptitle("Training data")
plt.subplot(311)
plt.plot(t_train_data, u_train)
plt.plot(t_train_data, np.array(predicted_y[:, 0]))
plt.subplot(312)
plt.plot(t_train_data, y_train[:, 0])
plt.plot(t_train_data, np.array(predicted_y[:, 1]))
plt.subplot(313)
plt.plot(t_train_data, y_train[:, 1])
plt.plot(t_train_data, np.array(predicted_y[:, 2]))


plt.figure(4)
plt.title("Loss per iteration")
plt.plot(range(n_iters + 1), loss_history)


# ############## Test on bew data-set for model validation ################
t_final = 300
num_test_data = int(t_final/dt)
t_process = np.linspace(0, t_final-dt, num=num_test_data)
u_test = pseudo_random_pulse(num_test_data, int(num_test_data/20), 100, 90, 110)
t_test = np.linspace(0, num_test_data-1, num=len(u_test))
# T, y_test, x_test = c.forced_response(G, T=t_test, U=u_test, X0=1)
t_, y_test = simulate_process(f=p.ode_model, t=t_test, u=u_test, x0=z0)

scaler_u.fit(u_test)
scaler_y.fit(y_test)

u_test_scaled = scaler_u.transform(u_test)
y_test_scaled = scaler_y.transform(y_test)

u0_test = tf.cast(u_test_scaled[0], dtype=tf.float32)
x0_test = tf.reshape(tf.cast([y_test_scaled[0]], dtype=tf.float32), [2, ])

neural_ode_test = NeuralODE(model, t=t_test)
init_state_test = tf.concat([u0_test, x0_test, y0_p], axis=0)

state_final_test, predicted_y_test = neural_ode_test.forward(init_state_test, u_test_scaled, return_states="numpy")

plt.figure(5)
plt.suptitle("Validation data")
plt.subplot(311)
plt.plot(t_test, u_test_scaled)
plt.plot(t_test, np.array(predicted_y_test[:, 0]))
plt.subplot(312)
plt.plot(t_test, y_test_scaled[:, 0])
plt.plot(t_test, np.array(predicted_y_test[:, 1]))
plt.subplot(313)
plt.plot(t_test, y_test_scaled[:, 1])
plt.plot(t_test, np.array(predicted_y_test[:, 2]))


print('Model Weights: ', model.weights)
#print('Estimated Kp: ', (max_scale_y/max_scale_u)*model.weights[0].numpy())
#print('Estimated Tau: ', model.weights[1].numpy())

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
import control as c
from neural_ode_u import NeuralODE

tf.executing_eagerly()


def get_batch(x, start, batch_length):
    x_batch = x[int(start):int(start)+batch_length]
    return x_batch


def pseudo_random_pulse(num_value, period, min_val, max_val):
    sig = np.zeros(num_value)
    j = 0
    new_val = np.clip(np.random.rand(), min_val, max_val)
    for k in range(num_value):
        sig[k] = new_val
        j = j + 1
        if j == period:
            new_val = np.clip(np.random.rand(), min_val, max_val)
            j = 0

    return sig


class Process:
    def __init__(self):
        super(Process, self).__init__()
        self.Kp = 2.0
        self.Tau = 10

    def ode_model(self, t, y, u):
        y_dot = (self.Kp*u - y)/self.Tau
        return y_dot


def simulate_process(f, t, u, x0):
    r = ode(f).set_integrator('zvode', method='bdf')
    xk = x0
    t_values = [t[0]]
    x_values = [xk]

    deltas_t = t[1:] - t[:-1]
    for it in range(len(deltas_t)):
        r.set_initial_value(xk, t[it]).set_f_params(u[it])
        xk = np.real(r.integrate(r.t + deltas_t[it]))[0]
        if r.successful():
            t_values.append(r.t)
            x_values.append(xk)

    return t_values, x_values


Kp = 2.0
Tau = 10.0
D = 15.0
[numD,denD] = c.pade(D,3)
G=c.tf( Kp , [ Tau , 1 ])

print(G)

A = -1/Tau
B = Kp/Tau
C = 1
D = 0
sys = c.StateSpace(A, B, C, D)

num_train_data = 3000
u_process = pseudo_random_pulse(num_train_data, 100, 0.02, 1.0)
t_process = np.linspace(0, num_train_data-1, num=len(u_process))
p = Process()
t_, y_process = simulate_process(f=p.ode_model, t=t_process, u=u_process, x0=0.5)

# T, y_process, xout = c.forced_response(sys, T=t_process, U=u_process, X0=0.5)
y_random = y_process + np.random.normal(0, 0.0, len(y_process))

max_scale_y = max(y_process)
max_scale_u = max(u_process)
t_train_data = t_process
u_train = u_process/max_scale_u
y_train = tf.cast(y_random/max_scale_y, dtype=tf.float32)

u0 = tf.cast(u_train[0], dtype=tf.float32)
y0 = y_train[0]

init_state = tf.concat([[u0], [y0]], axis=0)
# init_state = tf.cast([x0], dtype=tf.float32)


class FOPTD(tf.keras.Model):
    def __init__(self):
        super(FOPTD, self).__init__()
        self.Kp = tf.Variable(tf.cast(1, dtype=tf.float32))
        self.Tau = tf.Variable(tf.cast(30, dtype=tf.float32))

    def call(self, inputs, **kwargs):
        t, states = inputs
        u = states[0]
        y = states[1]
        y_dot = (self.Kp*u - y)/self.Tau
        u_dot = tf.cast(0, dtype=tf.float32)
        h = tf.concat([[u_dot], [y_dot]], axis=0)
        return h


# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)

model = FOPTD()
#for layer in model.layers: print(layer.get_config(), layer.get_weights())

batch_size = 300
t_sim = np.linspace(0, batch_size-1, num=batch_size)

neural_ode = NeuralODE(model, t=t_sim)


def compute_gradients_and_update_path(initial_state, u_target, y_target):
    with tf.GradientTape() as g:
        final_y, y_history = neural_ode.forward(initial_state, u_target, return_states="tf")
        predicted_path = tf.stack(y_history)  # -> (batch_time, batch_size, 2)
        prediction_loss = tf.reduce_sum(tf.square(predicted_path - y_target), axis=0)

    # back-propagate through solver with tensorflow
    gradients = g.gradient(prediction_loss, model.weights)
    optimizer.apply_gradients(zip(gradients, model.weights))
    return prediction_loss, predicted_path.numpy()

# Compile EAGER graph to static (this will be much faster)
compute_gradients_and_update = tf.function(compute_gradients_and_update_path)

# ############## Training ################
n_iters = 500
idx = np.arange(len(y_train) - batch_size, dtype=int)
batch_starts = np.random.choice(idx, size=n_iters + 1, replace=(n_iters + 1 > len(idx)))
tic = time.time()
loss_history = []


y_train_batch = get_batch(y_train, batch_starts[0], batch_size)
u_train_batch = get_batch(u_train, batch_starts[0], batch_size)

plt.close('all')
fig = plt.figure(1)
ax1 = fig.add_subplot(211)
line1, = ax1.plot(range(batch_size), u_train_batch)
ax1.set_ylim((min(u_train), max(u_train)))
ax2 = fig.add_subplot(212)
line2, = ax2.plot(range(batch_size), y_train_batch.numpy())
line3, = ax2.plot(range(batch_size), [0]*batch_size)
ax2.set_ylim((min(y_train.numpy()), max(y_train.numpy())))

for step in range(n_iters + 1):
    y_train_batch = get_batch(y_train, batch_starts[step], batch_size)
    u_train_batch = get_batch(u_train, batch_starts[step], batch_size)
    u0_batch = tf.cast(u_train_batch[0], dtype=tf.float32)
    y0_batch = y_train_batch[0]
    init_state_batch = tf.concat([[u0_batch], [y0_batch]], axis=0)
    loss, y_predicted = compute_gradients_and_update_path(init_state_batch, u_train_batch, y_train_batch)
    loss_history.append(loss.numpy())

    line1.set_ydata(u_train_batch)
    line2.set_ydata(y_train_batch.numpy())
    line3.set_ydata(y_predicted)
    fig.canvas.draw()
    fig.canvas.flush_events()

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


plt.figure(2)
plt.suptitle("Training data")
plt.subplot(211)
plt.plot(t_train_data, max_scale_y*u_train)
plt.subplot(212)
plt.plot(t_train_data, max_scale_y*y_train)
plt.plot(t_train_data, max_scale_y*np.array(predicted_y))


plt.figure(3)
plt.title("Loss per iteration")
plt.plot(range(n_iters + 1), loss_history)


# ############## Test on bew data-set for model validation ################
num_test_data = 1000
u_test = pseudo_random_pulse(num_test_data, 100, 0.0, 1.0)
t_test = np.linspace(0, num_test_data-1, num=len(u_test))
# T, y_test, x_test = c.forced_response(G, T=t_test, U=u_test, X0=1)
t_, y_test = simulate_process(f=p.ode_model, t=t_test, u=u_test, x0=0.5)

y_test_scaled = y_test/max_scale_y
u_test_scaled = u_test/max_scale_u

u0_test = tf.cast([u_test_scaled[0]], dtype=tf.float32)
x0_test = tf.cast([y_test_scaled[0]], dtype=tf.float32)

neural_ode_test = NeuralODE(model, t=t_test)
init_state_test = tf.concat([u0_test, x0_test], axis=0)
state_final_test, predicted_y_test = neural_ode_test.forward(init_state_test, u_test_scaled, return_states="numpy")

plt.figure(4)
plt.suptitle("Validation data")
plt.subplot(211)
plt.plot(t_test, u_test)

plt.subplot(212)
plt.plot(t_test, y_test)
plt.plot(t_test, max_scale_y*np.array(predicted_y_test))

print('Estimated Kp: ', (max_scale_y/max_scale_u)*model.weights[0].numpy())
print('Estimated Tau: ', model.weights[1].numpy())

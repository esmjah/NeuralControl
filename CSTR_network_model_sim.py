import numpy as np
import numpy.random as npr
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow.contrib.eager as tfe
import pandas
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
# from keras.layers import Dropout
from keras.layers import Flatten


# from sklearn import preprocessing
def data_normalizer(input_data):
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(input_data)
    input_data_normalized = pandas.DataFrame(np_scaled)
    input_data_n = input_data_normalized
    input_data_n = np.array(input_data_n.loc[:, :])
    return input_data_n


# '''''''''''''''''''''''''''''''''''''''''''''''''
dataframe1 = pandas.read_excel(open('.\CSTR_TEST_V1_train.xlsx', 'rb'), sheet_name='Sheet1')

dataset1 = dataframe1.values

total_data1 = dataset1.astype('float32')
total_data_orig = np.copy(total_data1)
total_data1 = data_normalizer(total_data1)

total_data = total_data1
# total_data=np.concatenate((total_data1,total_data2,total_data3), axis=0)

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# total_data=data_normalizer(total_data)
ts = 0
tfi = 1000
time = total_data_orig[:, 0]
# time=total_data[0:600,0]
# time=total_data[ts:tfi,0]
# time=np.linspace(0.0, 30.03, num=total_data.shape[0])
ds = time.shape[0]
print(ds)

output_data_test1 = total_data[ts:ds, [1, 2, 3]]

plt.figure(1)
plt.subplot(311)
plt.title("Training data")
plt.plot(time, output_data_test1[:, 0], label="x_1")
plt.legend()
plt.subplot(312)
plt.plot(time, output_data_test1[:, 1], label="x_2")
plt.legend()
plt.subplot(313)
plt.plot(time, output_data_test1[:, 2], label="u")
plt.legend()
plt.show()

t_grid = time

# plt.plot(time,output_data_test1[:,2])
# plt.plot(time,input_data_test1)
keras = tf.keras
tf.enable_eager_execution()

from neural_ode import NeuralODE

data_size = ds
# data_size = 1001
batch_time = 3  # this seems to works the best ...
niters = 1000
batch_size = 150

# true_y0 = tf.to_float([[0.0200024]])
true_y0 = tf.to_float([[output_data_test1[0, :]]])
# true_y0 = tf.to_float([[0.0200024,0.979998]])
# true_y=tf.to_float([output_data_test1])
true_y = output_data_test1

print(true_y0)
print(true_y.shape)


# Create batch generator

# Sample frament of trajectory, here we want to start from some random position y0, and then force model to match final posiion yN

def get_path_batch():
    starts = np.random.choice(np.arange(data_size - batch_time - 1, dtype=np.int64), batch_size, replace=False)
    #    starts = np.random.choice(np.arange(data_size - batch_time - 1, dtype=np.int64), batch_size, replace=True)
    #    starts = np.array([0,52,103])
    batch_y0 = true_y[starts]  # (batch_size, 2)
    batch_yN = [true_y[starts + i] for i in range(batch_time)]  # (batch_time, batch_size, 2)
    return tf.cast(batch_y0, float), tf.cast(batch_yN, float)


# simple network which is used to learn trajectory
class ODEModel(tf.keras.Model):
    #    model = Sequential()
    #    model.add(Dense(100, input_dim=4,activation= 'tanh'))
    #    model.add(Flatten())

    # mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
    def __init__(self):
        super(ODEModel, self).__init__()
        self.linear1 = keras.layers.Dense(50, activation="tanh", name='dense_1')
        self.linear2 = keras.layers.Dense(50, activation="tanh", name='dense_2')
        #        self.linear3 = keras.layers.Dense(500, activation="tanh",name='dense_3')
        #        self.linear4 = keras.layers.Dense(500, activation="tanh",name='dense_4')
        #        self.linear5 = keras.layers.Dense(500, activation="tanh",name='dense_5')
        #        self.linear6 = keras.layers.Dense(70, activation="relu")
        #        self.linear7 = keras.layers.Dense(60, activation="relu")
        #        self.linear8 = keras.layers.Dense(50, activation="relu")
        #        self.linear9 = keras.layers.Dense(50, activation="relu")
        #        self.linear10 = keras.layers.Dense(40, activation="relu")
        self.linear3 = keras.layers.Dense(3, name='dense_3')

    def call(self, inputs, **kwargs):
        t, y = inputs
        h = y
        h = self.linear1(h)
        h = self.linear2(h)
        h = self.linear3(h)
        #        h = self.linear4(h)
        #        h = self.linear5(h)
        #        h = self.linear6(h)
        #        h = self.linear7(h)
        #        h = self.linear8(h)
        #        h = self.linear9(h)
        #        h = self.linear10(h)
        #        h = self.linear11(h)
        return h


##mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
# NeuralODE integrator with RK4 solver


model = ODEModel()
# for layer in model.layers: print(layer.get_config(), layer.get_weights())

neural_ode = NeuralODE(model, t=t_grid[:batch_time])
neural_ode_test = NeuralODE(model, t=t_grid)

# optimizer = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.95)
# optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4)
optimizer = tf.train.AdamOptimizer()


def compute_gradients_and_update_path(batch_y0, batch_yN):
    with tf.GradientTape() as g:
        pred_y, y_points = neural_ode.forward(batch_y0, return_states="tf")
        pred_path = tf.stack(y_points)  # -> (batch_time, batch_size, 2)
        loss = tf.reduce_mean(tf.abs(pred_path - batch_yN), axis=1)  # -> (batch_time, 2)
        loss = tf.reduce_mean(loss, axis=0)

    # backpropagate through solver with tensorflow
    gradients = g.gradient(loss, model.weights)
    optimizer.apply_gradients(zip(gradients, model.weights))
    return loss


compute_gradients_and_update_path = tfe.defun(compute_gradients_and_update_path)

loss_history = []
for step in tqdm(range(niters + 1)):
    batch_y0, batch_yN = get_path_batch()
    loss = compute_gradients_and_update_path(batch_y0, batch_yN)
    loss_history.append(loss.numpy())

    if step % 500 == 0:
        yN, states_history_model = neural_ode_test.forward(true_y0, return_states="numpy")
    #        plot_spiral([true_y, np.concatenate(states_history_model)])
#        plt.show()
predicted_y = np.concatenate(states_history_model)
# states=np.array([true_y,predicted_y])
# plt.figure(1)
# plt.ylim(0, 0.1)
# plt.plot(loss_history)
# plt.show
#
# # plt.figure
# plt.figure(2)
# plt.figure(figsize=(10, 6))
# plt.subplot(311)
# plt.plot(t_grid, true_y[:, 0], t_grid, predicted_y[:, 0][:, 0])
# plt.subplot(312)
# plt.plot(t_grid, true_y[:, 1], t_grid, predicted_y[:, 0][:, 1])
# plt.subplot(313)
# plt.plot(t_grid, true_y[:, 2], t_grid, predicted_y[:, 0][:, 2])
# plt.show
# # plt.close()

# %%-------------------------------------------------- EXPLORATION --------------------------------
dataframeE = pandas.read_excel(open('.\CSTR_TEST_V1_test.xlsx', 'rb'), sheet_name='Sheet1')
# dataframeE = pandas.read_excel(open('.\CSTR_TEST_V1_tes.xlsx', 'rb'), sheet_name='Sheet1')
datasetE = dataframeE.values
total_dataE = datasetE.astype('float32')

total_dataE = data_normalizer(total_dataE)

# output_data_test1E=total_dataE[ts:ds,[1,2,3]]
output_data_test1E = total_dataE[:, [1, 2, 3]]

true_y0E = tf.cast([[output_data_test1E[0, :]]], float)
true_yE = output_data_test1E

time = total_dataE[:, 0]
# t_gridE=np.linspace(0.0, 1.0, num=true_yE[:,0].shape[0])
t_gridE = time
# t=np.arange(0,4,0.01)
t = t_gridE
true_yNE = tf.cast([true_yE[0]], float)
# t=total_data[100:400,0]
neural_ode_extrapolation = NeuralODE(model, t)
yNE, states_history_modelE = neural_ode_extrapolation.forward(true_yNE, return_states="numpy")
predicted_yE = np.concatenate(states_history_modelE)
# plt.figure
# plt.figure(3)
# plt.figure(figsize=(10, 6))
# plt.subplot(411)
# plt.plot(t, true_yE[:, 0], t, predicted_yE[:, 0])
# plt.subplot(412)
# plt.plot(t, true_yE[:, 1], t, predicted_yE[:, 1])
# plt.subplot(413)
# plt.plot(t, true_yE[:, 2], t, predicted_yE[:, 2])
# # plt.subplot(414)
# # plt.plot(t,true_yE[:,3],t,predicted_yE[:,3])
# plt.show
#
# plt.figure(4)
# plt.subplot(211)
# plt.plot(t_grid, true_y[:, 0], t_grid, predicted_y[:, 0][:, 0])
# plt.subplot(212)
# plt.plot(t, true_yE[:, 0], t, predicted_yE[:, 0])
# plt.show

print('error=  ', true_yE[0:6, 0] - predicted_yE[0:6, 0])

print('mean=  ', np.mean(abs(true_yE[0:6, 0] - predicted_yE[0:6, 0])))

# %%
model.save_weights('my_model_weights.h5')

#
# model = Sequential()
# model.add(Dense(400, input_dim=3,activation= 'tanh', name='dense_1'))  # will be loaded
# model.add(Dense(10, name='new_dense'))
#
#
#
# for layer in model.layers: print(layer.get_config(), layer.get_weights())


# new stuff written by Esmaeil
Ts = 0.01  # Sampling interval of the simulation loop
t_iter = np.linspace(0, Ts, num=11)  # Divide Ts to 10 smaller steps for better accuracy of ODE solder
x0 = true_yE[0]
x_k = x0
neural_ode_extrapolation = NeuralODE(model, t_iter)
T_sim = 4.4  # final simulation time
N_sim = int(T_sim/Ts)  # Number of iterations of the simulation loop
x_process = np.array([x0])
t_process = np.linspace(0, T_sim, N_sim + 1)  # Simulate the process for 4 minutes.
Nu = int(N_sim/10)
u_process = (Nu+1)*[0] + Nu*[0.25] + Nu*[0.5] + Nu*[0.75] + Nu*[1] + Nu*[0] + Nu*[0.25] + Nu*[0.5] + Nu*[0.75] + Nu*[1]
for k in tqdm(range(N_sim)):
    x_k_ = np.array([x_k[0], x_k[1], u_process[k]])
    yNE, states_temp = neural_ode_extrapolation.forward(tf.cast([x_k_], float), return_states="numpy")
    x_temp = np.concatenate(states_temp)
    x_k = x_temp[-1]
    x_process = np.append(x_process, np.array([x_k]), axis=0)

x_1 = x_process[:, 0]
x_2 = x_process[:, 1]
u = x_process[:, 2]

plt.figure(5)
plt.subplot(311)
plt.title("Network simulation")
plt.plot(t_process, x_1, label="x_1")
plt.legend()
plt.subplot(312)
plt.plot(t_process, x_2, label="x_2")
plt.legend()
plt.subplot(313)
plt.plot(t_process, u, label="u")
plt.legend()
plt.show()


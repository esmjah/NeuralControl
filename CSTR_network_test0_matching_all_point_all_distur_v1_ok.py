import numpy as np
import numpy.random as npr
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow.contrib.eager as tfe
import pandas
from sklearn import preprocessing

#from sklearn import preprocessing
def data_normalizer(input_data):
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(input_data)
    input_data_normalized = pandas.DataFrame(np_scaled)
    input_data_n=input_data_normalized
    input_data_n=np.array(input_data_n.loc[:,:])
    return input_data_n

#'''''''''''''''''''''''''''''''''''''''''''''''''
dataframe1 = pandas.read_excel(open('.\CSTR_TEST_V1_train.xlsx', 'rb'), sheet_name='Sheet1')

dataset1 = dataframe1.values

total_data1 = dataset1.astype( 'float32' )
total_data_orig=np.copy(total_data1)
total_data1=data_normalizer(total_data1)

total_data=total_data1
#total_data=np.concatenate((total_data1,total_data2,total_data3), axis=0)

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#total_data=data_normalizer(total_data)
ts=0
tfi=1000
time=total_data[:,0]
#time=total_data[0:600,0]
#time=total_data[ts:tfi,0]
#time=np.linspace(0.0, 30.03, num=total_data.shape[0])
ds=time.shape[0]
print(ds)

output_data_test1=total_data[ts:ds,[1,2,3]]

t_grid=time
plt.plot(time,output_data_test1[:,0])
plt.plot(time,output_data_test1[:,1])
#plt.plot(time,output_data_test1[:,2])
#plt.plot(time,input_data_test1)
keras = tf.keras
tf.enable_eager_execution()

from neural_ode import NeuralODE

data_size = ds
#data_size = 1001
batch_time = 3 # this seems to works the best ...
niters = 1000
batch_size = 8

#true_y0 = tf.to_float([[0.0200024]])
true_y0 = tf.to_float([[output_data_test1[0,:]]])
#true_y0 = tf.to_float([[0.0200024,0.979998]])
#true_y=tf.to_float([output_data_test1])
true_y = output_data_test1

print(true_y0)
print(true_y.shape)



#Create batch generator

#Sample frament of trajectory, here we want to start from some random position y0, and then force model to match final posiion yN

def get_path_batch():
    starts = np.random.choice(np.arange(data_size - batch_time - 1, dtype=np.int64), batch_size, replace=False)
#    starts = np.random.choice(np.arange(data_size - batch_time - 1, dtype=np.int64), batch_size, replace=True)
#    starts = np.array([0,52,103])
    batch_y0 = true_y[starts] # (batch_size, 2)
    batch_yN = [true_y[starts + i] for i in range(batch_time)] # (batch_time, batch_size, 2)
    return tf.to_float(batch_y0), tf.to_float(batch_yN)


# simple network which is used to learn trajectory
class ODEModel(tf.keras.Model):
    
    def __init__(self):
        super(ODEModel, self).__init__()
        self.linear1 = keras.layers.Dense(400, activation="tanh")
        self.linear2 = keras.layers.Dense(400, activation="tanh")
        self.linear3 = keras.layers.Dense(400, activation="tanh")
#        self.linear4 = keras.layers.Dense(100, activation="relu")
#        self.linear5 = keras.layers.Dense(80, activation="relu")
#        self.linear6 = keras.layers.Dense(70, activation="relu")
#        self.linear7 = keras.layers.Dense(60, activation="relu")
#        self.linear8 = keras.layers.Dense(50, activation="relu")
#        self.linear9 = keras.layers.Dense(50, activation="relu")
#        self.linear10 = keras.layers.Dense(40, activation="relu")
        self.linear4 = keras.layers.Dense(3)        

    def call(self, inputs, **kwargs):
        t, y = inputs
        h = y
        h = self.linear1(h)
        h = self.linear2(h)
        h = self.linear3(h)
        h = self.linear4(h)
#        h = self.linear5(h)
#        h = self.linear6(h)
#        h = self.linear7(h)
#        h = self.linear8(h)
#        h = self.linear9(h)
#        h = self.linear10(h)
#        h = self.linear11(h)
        return h
    
#NeuralODE integrator with RK4 solver


model = ODEModel()
neural_ode = NeuralODE(model, t = t_grid[:batch_time])
neural_ode_test = NeuralODE(model, t=t_grid)

#optimizer = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.95)
#optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4)
optimizer = tf.train.AdamOptimizer()


def compute_gradients_and_update_path(batch_y0, batch_yN):
    with tf.GradientTape() as g:
        
        pred_y, y_points = neural_ode.forward(batch_y0, return_states="tf")        
        pred_path = tf.stack(y_points)  # -> (batch_time, batch_size, 2)        
        loss = tf.reduce_mean(tf.abs(pred_path - batch_yN), axis=1) # -> (batch_time, 2)   
        loss = tf.reduce_mean(loss, axis=0)
    
    # backpropagate through solver with tensorflow
    gradients = g.gradient(loss, model.weights)  
    optimizer.apply_gradients(zip(gradients, model.weights))
    return loss

compute_gradients_and_update_path = tfe.defun(compute_gradients_and_update_path)

loss_history = []
for step in tqdm(range(niters+1)):
    batch_y0, batch_yN = get_path_batch()
    loss = compute_gradients_and_update_path(batch_y0, batch_yN)    
    loss_history.append(loss.numpy())
    
    if step % 500 == 0:        
        yN, states_history_model = neural_ode_test.forward(true_y0, return_states="numpy")        
#        plot_spiral([true_y, np.concatenate(states_history_model)])        
#        plt.show()
predicted_y=np.concatenate(states_history_model)
#states=np.array([true_y,predicted_y])
plt.figure(1)
plt.ylim(0,0.1)
plt.plot(loss_history)
plt.show

#plt.figure
plt.figure(2)
plt.figure(figsize=(10,6))
plt.subplot(311)
plt.plot(t_grid,true_y[:,0],t_grid,predicted_y[:,0][:,0])
plt.subplot(312)
plt.plot(t_grid,true_y[:,1],t_grid,predicted_y[:,0][:,1])
plt.subplot(313)
plt.plot(t_grid,true_y[:,2],t_grid,predicted_y[:,0][:,2])
plt.show
#plt.close()

#%%-------------------------------------------------- EXPLORATION --------------------------------
dataframeE = pandas.read_excel(open('.\CSTR_TEST_V1_test.xlsx', 'rb'), sheet_name='Sheet1')
#dataframeE = pandas.read_excel(open('.\CSTR_TEST_V1_tes.xlsx', 'rb'), sheet_name='Sheet1')
datasetE = dataframeE.values
total_dataE = datasetE.astype( 'float32' )


total_dataE=data_normalizer(total_dataE)

#output_data_test1E=total_dataE[ts:ds,[1,2,3]]
output_data_test1E=total_dataE[:,[1,2,3]]

true_y0E = tf.cast([[output_data_test1E[0,:]]],float)
true_yE = output_data_test1E

time=total_dataE[:,0]
#t_gridE=np.linspace(0.0, 1.0, num=true_yE[:,0].shape[0])
t_gridE=time
#t=np.arange(0,4,0.01)
t=t_gridE
true_yNE = tf.cast([true_yE[0]],float)
#t=total_data[100:400,0]
neural_ode_extrapolation = NeuralODE(model, t)
yNE, states_history_modelE = neural_ode_extrapolation.forward(true_yNE, return_states="numpy")
predicted_yE=np.concatenate(states_history_modelE)
#plt.figure
plt.figure(3)
plt.figure(figsize=(10,6))
plt.subplot(411)
plt.plot(t,true_yE[:,0],t,predicted_yE[:,0])
plt.subplot(412)
plt.plot(t,true_yE[:,1],t,predicted_yE[:,1])
plt.subplot(413)
plt.plot(t,true_yE[:,2],t,predicted_yE[:,2])
#plt.subplot(414)
#plt.plot(t,true_yE[:,3],t,predicted_yE[:,3])
plt.show
        
plt.figure(4)        
plt.subplot(211)
plt.plot(t_grid,true_y[:,0],t_grid,predicted_y[:,0][:,0])        
plt.subplot(212)
plt.plot(t,true_yE[:,0],t,predicted_yE[:,0])        
plt.show


print('error=  ',true_yE[0:6,0]-predicted_yE[0:6,0])

print('mean=  ',np.mean(true_yE[0:6,0]-predicted_yE[0:6,0]))

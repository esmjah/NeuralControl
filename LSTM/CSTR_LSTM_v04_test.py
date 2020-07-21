import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from tqdm import tqdm
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout  
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import tensorflow as tf
from numpy.random import seed
from scipy import io
import time
from scipy.integrate import ode
seed(0)
tf.set_random_seed(42)

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return numpy.array(dataX), numpy.array(dataY)

def data_normalizer(input_data):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    np_scaled = min_max_scaler.fit_transform(input_data)
    data_max = min_max_scaler.data_max_
    data_min = min_max_scaler.data_min_
    input_data_normalized = pandas.DataFrame(np_scaled)
    input_data_n=input_data_normalized
    input_data_n=np.array(input_data_n.loc[:,:])
    return input_data_n, data_min, data_max


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
# load the train data


data = io.loadmat(file_name='cstr_data_1.mat')

dt = data['dt'][0][0]
z0 = data['z0'][0]
t_process = data['t_process'][0]
u_process = data['u_process']
y_process = data['y_process']

train=np.concatenate((np.array(y_process), np.array(u_process)),axis=1)





#------------------------------------------------------
# normalize the dataset
trainX=train
trainX, min_orig, max_orig =data_normalizer(train)
trainp, min_orig, max_orig =data_normalizer(train)
min_orig_PID=min_orig
max_orig_PID=max_orig

col_=train.shape[1]
# load the test data
look_back = 5
no_of_features=3  # Ca, T ,u
dataframe = pandas.read_csv(r'.\CSTR_TEST_V1_test.csv')
dataset = dataframe.values
test1 = dataset.astype( 'float32' )
test=test1[:,[1,2,3]]

t_test = data['t_test'][0]
u_test = data['u_test']
y_test = data['y_test']

test=np.concatenate((np.array(y_test), np.array(u_test)),axis=1)

#----------------------
testX0, testY0 = create_dataset(test, look_back)

x0_original = testX0[0:1,:,:]

testX, min_orig, max_orig =data_normalizer(test)
testp, min_orig, max_orig =data_normalizer(test)
#

trainX, trainY = create_dataset(trainX, look_back)
testX, testY = create_dataset(testX, look_back)

#%111111111111111111111111111111
# create and fit the LSTM network
model = Sequential()
#model.add(LSTM(150, input_dim=look_back,return_sequences=True))
model.add(LSTM(150, input_shape=(look_back, no_of_features),return_sequences=True))


model.add(Dropout(0.2))  

model.add(LSTM(150, input_dim=look_back,return_sequences=True))
model.add(Dropout(0.2))  

model.add(LSTM(150, input_dim=look_back,return_sequences=True))
model.add(Dropout(0.2))  

model.add(LSTM(100, input_dim=look_back,return_sequences=True))
model.add(Dropout(0.2)) 

model.add(LSTM(100, input_dim=look_back,return_sequences=True))
model.add(Dropout(0.2))  

model.add(LSTM(100, input_dim=look_back,return_sequences=True))
model.add(Dropout(0.2))  
 
model.add(LSTM(80, input_dim=look_back,return_sequences=True))
model.add(Dropout(0.2))  


model.add(LSTM(80, input_dim=look_back,return_sequences=True))
model.add(Dropout(0.2)) 

#model.add(LSTM(150, input_dim=look_back))
model.add(LSTM(80, input_shape=(look_back, no_of_features)))
model.add(Dropout(0.2))  


model.add(Dense(col_))


opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss= 'mean_squared_error' , optimizer=opt )
#model.compile(loss= 'mean_squared_error' , optimizer='adam' )
#model.compile(loss= 'mean_squared_error' , optimizer='sgd' )


#tic = time.time()
#model.fit(trainX, trainY, nb_epoch=10000, batch_size=32, verbose=2)
#toc = time.time()
#training_time = toc - tic
#print("Training time: ", training_time)
# make predictions
model.load_weights("CSTR_SecondOrder_LSTM_v3_1.ckpt")

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
#testPredict1 = model.predict(testX[0:1,:,:])
#testX[0:1,:,:]

## invert predictions
#trainPredict = scaler.inverse_transform(trainPredict)
#trainY = scaler.inverse_transform([trainY])
#testPredict = scaler.inverse_transform(testPredict)
#testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:,0]))
trainScore = math.sqrt(mean_squared_error(trainY[:,1], trainPredict[:,1]))
testScore = math.sqrt(mean_squared_error(testY[:,1], testPredict[:,1]))
trainScore = math.sqrt(mean_squared_error(trainY[:,2], trainPredict[:,2]))
testScore = math.sqrt(mean_squared_error(testY[:,2], testPredict[:,2]))
print( 'Train Score0: %.2f RMSE' % (trainScore))
print( 'Train Score1: %.2f RMSE' % (trainScore))
print( 'Train Score2: %.2f RMSE' % (trainScore))

print( 'Test Score1: %.2f RMSE' % (testScore))
print( 'Test Score0: %.2f RMSE' % (testScore))
print( 'Test Score2: %.2f RMSE' % (testScore))

loss_ = np.sum(np.square(trainPredict[:,0] - trainY[:,0]), axis=0)
training_loss = np.sum(loss_)
print("Training loss: ", training_loss)


dataset=numpy.concatenate((trainp, testp), axis=0)



trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions


#plt.plot(scaler.inverse_transform(dataset))
#plt.plot(trainPredictPlot)
#plt.plot(testPredictPlot)

plt.figure(1)
#plt.figure(figsize=(18,13))  
plt.plot(dataset[:,0], color='blue', label='Actual Time Series Value')  
plt.plot(trainPredictPlot[:,0] , 'r--', label='Train Phase')  
plt.plot(testPredictPlot[:,0] , 'g--', label='Test Phase') 
plt.title('LSTM network train and test phase for $C_a$ ') 
plt.xlabel('Epochs')  
plt.ylabel('Function Value')  
plt.legend(loc="lower left") 


plt.figure(2)
#plt.figure(figsize=(18,13))  
plt.plot(dataset[:,1], color='blue', label='Actual Time Series Value')  
plt.plot(trainPredictPlot[:,1] , color='red', label='Train Phase')  
plt.plot(testPredictPlot[:,1] , color='green', label='Test Phase') 
plt.title('LSTM Network train and test phase for T ') 
plt.xlabel('Epochs')  
plt.ylabel('Function Value')  
plt.legend(loc="lower left") 

plt.figure(3)
#plt.figure(figsize=(18,13))  
plt.plot(dataset[:,2], color='blue', label='Actual Time Series Value')  
plt.plot(trainPredictPlot[:,2] , color='red', label='Train Phase')  
plt.plot(testPredictPlot[:,2] , color='green', label='Test Phase') 
plt.title('LSTM Network train and test phase for $q_c$ ') 
plt.xlabel('Epochs')  
plt.ylabel('Function Value')  
plt.legend(loc="lower left") 


dataset1=testp
#--------------------------------------------------

#plt.plot(y_test[:,1])

plt.show()
#---------------------------------------------------------
from matplotlib.font_manager import FontProperties
#fig, ax = plt.subplots(figsize=(18,13))
fig, ax = plt.subplots()
#fig = plt.figure()
plt.xlim(0,6200)
axins = ax.inset_axes([1.1, 0.50, 0.67, 0.67])
x1, x2, y1, y2 = 850.0, 1000.0, -0.5, 1.0
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')
ax.indicate_inset_zoom(axins)

ax.plot(dataset[:,0], color='blue', label='Actual Time Series Value')  
ax.plot(trainPredictPlot[:,0] , 'r--', label='Train Phase')  
ax.plot(testPredictPlot[:,0] , 'g--', label='Test Phase')
axins.plot(dataset[:,0], color='blue')  
axins.plot(trainPredictPlot[:,0] , color='red')  
axins.plot(testPredictPlot[:,0] , 'g--')


font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
fig.suptitle('LSTM Network output in train and test phase for $C_a$',fontweight='bold',fontproperties=font, fontsize=14)
plt.xlabel('epochs', fontproperties=font, fontsize=12)
plt.ylabel('$C_a$', fontproperties=font, fontsize=12)
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, mode="expand", borderaxespad=0.)
#plt.legend(bbox_to_anchor=(0.5, 0.5), loc='upper left', ncol=1, borderaxespad=0.)
plt.legend(loc="lower left")
fig.savefig('fig_lstm_zoom_1.jpg', dpi=600,bbox_inches='tight',pad_inches=0.5)
#------------------------------------------

fig, ax = plt.subplots()
#fig = plt.figure()
plt.xlim(0,6200)
axins = ax.inset_axes([1.1, 0.50, 0.67, 0.67])
x1, x2, y1, y2 = 850.0, 1000.0, -1.2, 1.2
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')
ax.indicate_inset_zoom(axins)

ax.plot(dataset[:,1], color='blue', label='Actual Time Series Value')  
ax.plot(trainPredictPlot[:,1] , 'r.-', label='Train Phase')  
ax.plot(testPredictPlot[:,1] , 'g--', label='Test Phase')
axins.plot(dataset[:,1], color='blue')  
axins.plot(trainPredictPlot[:,1] , color='red')  
axins.plot(testPredictPlot[:,1] , 'g--')


font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
fig.suptitle('LSTM Network output in train and test phase for T',fontweight='bold',fontproperties=font, fontsize=14)
plt.xlabel('epochs', fontproperties=font, fontsize=12)
plt.ylabel('$C_a$', fontproperties=font, fontsize=12)
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, mode="expand", borderaxespad=0.)
#plt.legend(bbox_to_anchor=(0.5, 0.5), loc='upper left', ncol=1, borderaxespad=0.)
plt.legend(loc="lower left")
fig.savefig('fig_lstm_zoom_2.jpg', dpi=600,bbox_inches='tight',pad_inches=0.5)
#--------------------------------------------------------
fig, ax = plt.subplots()
#fig = plt.figure()
plt.xlim(0,6200)
axins = ax.inset_axes([1.1, 0.50, 0.67, 0.67])
x1, x2, y1, y2 = 850.0, 1000.0, -1.2, -0.5
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')
ax.indicate_inset_zoom(axins)

ax.plot(dataset[:,2], color='blue', label='Actual Time Series Value')  
ax.plot(trainPredictPlot[:,2] , 'r.-', label='Train Phase')  
ax.plot(testPredictPlot[:,2] , 'g--', label='Test Phase')
axins.plot(dataset[:,2], color='blue')  
axins.plot(trainPredictPlot[:,2] , color='red')  
axins.plot(testPredictPlot[:,2] , 'g--')


font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
fig.suptitle('LSTM Network output in train and test phase for $q_c$',fontweight='bold',fontproperties=font, fontsize=14)
plt.xlabel('epochs', fontproperties=font, fontsize=12)
plt.ylabel('$C_a$', fontproperties=font, fontsize=12)
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, mode="expand", borderaxespad=0.)
#plt.legend(bbox_to_anchor=(0.5, 0.5), loc='upper left', ncol=1, borderaxespad=0.)
plt.legend(loc="lower left")
fig.savefig('fig_lstm_zoom_3.jpg', dpi=600,bbox_inches='tight',pad_inches=0.5)

#model.save_weights("CSTR_SecondOrder_LSTM_v3_2.ckpt")



#%%
##################### PID Controller ################################
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

test_pid=np.concatenate((np.array(y_process), np.array(u_process)),axis=1)
testX0_pid, testY0_pid = create_dataset(test_pid, look_back)

x0_original_pid = testX0_pid[0:1,:,:]





Ts = 0.01  # Sampling interval of the simulation loop
t_iter = np.linspace(0, Ts, num=11)  # Divide Ts to 10 smaller steps for better accuracy of ODE solver
# Scaling x0 to original process
#min_scale = min_orig_PID[0:3]
min_scale =  np.concatenate((min_y, min_u),axis=0)
#max_scale = max_orig_PID[0:3]
max_scale =  np.concatenate((max_y, max_u),axis=0)
x0 = x0_original_pid
x_k = x0
#x_k = z0

#neural_ode_extrapolation = NeuralODE(model, t_iter)
T_sim = 45  # final simulation time
N_sim = int(T_sim/Ts)  # Number of iterations of the simulation loop
#x_process = np.array([x0])
#x_process=test1[0,[1,2,3]]
x_process=test_pid[0,:]
x_process=x_process.reshape(1,3)
t_process = np.linspace(0, T_sim, N_sim + 1)  # Simulate the process for 4 minutes.

# Controller variables
#u0 =100.0 
u0=u_process[0]
#u0 = x0[2]
u_k_1 = u0
u_control = np.array([u0])
IntegralError = 0.0
Error_k_1 = IntegralError


# Controller parameters
Kc0 = 0.2   # 0.2 ok
Kc = scale_u*Kc0/scale_y[0]
Ti = 0.1  #  0.1 ok 
Td = 0.05 #  0.05 ok 
#setpoint = 0.085
setpoint_iter = x0[0,0,0]
setpoint_lstm = np.array([setpoint_iter])

uMin = u0 - scale_u
uMax = u0 + scale_u
du_max = scale_u * 0.1

#uk_PID_fig=np.array([u0])
uk_PID_fig=u0
x_k_=x_k
##for k in tqdm(range(N_sim)):
for k in tqdm(range(N_sim)):
    if k > N_sim/9:
        setpoint_iter = 0.09
    if k > N_sim/3:
        setpoint_iter = 0.08
    if k > N_sim/1.5:
        setpoint_iter = 0.085 
    setpoint_lstm = np.append(setpoint_lstm, setpoint_iter)
    Error_k = np.array([setpoint_iter - x_k[0,0,0]])
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
#    x_k_ = scale_data(np.array([x_k[0,0,0], x_k[0,0,1], u_k]), min_scale, max_scale)
    for ii in range(look_back):
        x_k_temp1 = scale_data(x_k[0,ii,0:2], y_process[0], scale_y)
        u_k_temp1 = scale_data(u_k, u_process[0], scale_u)
        x_k_[0,ii,:] = np.concatenate((x_k_temp1,u_k_temp1),axis=0)

    u_k_= u_k_temp1
    u_iter = np.repeat(u_k_, len(t_iter))
    u_iter.shape = (len(t_iter), 1)
    
    states_temp = model.predict(x_k_)
    x_temp = np.concatenate(states_temp)
#   
    
    # descale the NN output to original process dimensions
    x_k_temp12 = descale_data(np.array(x_temp[0:2]), y_process[0], scale_y)
    u_k_temp12 = descale_data(np.array([x_temp[2]]), u_process[0], scale_u)
    x_k_temp = np.concatenate((x_k_temp12,u_k_temp12),axis=0)

    x_k_temp=x_k_temp.reshape(1,3)
    x_process = np.append(x_process, x_k_temp, axis=0)
#    print('k=',k,'    x_process.shape',x_process.shape)
    x_process_s= x_process.shape[0]
    x_process=(x_process).reshape(x_process_s,3)
    x_k_temp2=x_k_temp.reshape(1,1,3)
    x_k_=x_k_[0,1:5,:]
    x_k_=x_k_.reshape(1,4,3)
#    x_k_[0,4,:]=x_k_temp2
    x_k_=np.append(x_k_,x_k_temp2, axis=1)

x_1 = x_process[:, 0]
x_2 = x_process[:, 1]
u = x_process[:, 2]



#plt.show()




#%% ******************* Simulation of real process with PID **********************
p = Process()
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
    if k > N_sim/9:
        setpoint_iter = 0.09
    if k > N_sim/3:
        setpoint_iter = 0.08
    if k > N_sim/1.5:
        setpoint_iter = 0.085 
#    if k > N_sim/4:
#        setpoint_iter = 0.09
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
plt.plot(t_process, x_1, linestyle='dashed',  label="x_1 LSTM")
plt.plot(t_process, setpoint_process, label="setpoint")
plt.legend(loc='best')
plt.subplot(312)
plt.plot(t_process, x_2_process,  label="x_2 process")
plt.plot(t_process, x_2, linestyle='dashed', label="x_2 LSTM")
plt.legend(loc='best')
plt.subplot(313)
plt.plot(t_process, u_PID_process, label="u process")
plt.plot(t_process, u, linestyle='dashed', label="u LSTM")
plt.legend(loc='best')
plt.savefig('output.png', dpi=600, format='png')



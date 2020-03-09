import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout  
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
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

# load the train data
#dataframe = pandas.read_csv(r'.\train1.csv')
dataframe = pandas.read_csv(r'.\CSTR_TEST_V1_train.csv')
dataset = dataframe.values
train1 = dataset.astype( 'float32' )
train=train1[:,[1,2,3]]
# normalize the dataset
trainX, min_orig, max_orig =data_normalizer(train)
trainp, min_orig, max_orig =data_normalizer(train)

col_=train.shape[1]
# load the test data
#dataframe = pandas.read_csv(r'.\test1.csv')
dataframe = pandas.read_csv(r'.\CSTR_TEST_V1_test.csv')
dataset = dataframe.values
test1 = dataset.astype( 'float32' )
test=test1[:,[1,2,3]]
# normalize the dataset
testX, min_orig, max_orig =data_normalizer(test)
testp, min_orig, max_orig =data_normalizer(test)
#
look_back = 5
trainX, trainY = create_dataset(trainX, look_back)
testX, testY = create_dataset(testX, look_back)
#print(trainX)
#print(trainY)


# reshape input to be [samples, time steps, features]

trainX = numpy.reshape(trainX, (trainX.shape[0], col_, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], col_, testX.shape[1]))

#%%111111111111111111111111111111
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(150, input_dim=look_back,return_sequences=True))
model.add(Dropout(0.2))  

#model.add(LSTM(20, input_dim=look_back,return_sequences=True))
#model.add(Dropout(0.2))  

model.add(LSTM(150, input_dim=look_back))
model.add(Dropout(0.2))  

model.add(Dense(col_))


model.compile(loss= 'mean_squared_error' , optimizer='adam' )
#model.compile(loss= 'mean_squared_error' , optimizer='sgd' )

#model.fit(trainX, trainY, nb_epoch=100, batch_size=32, verbose=2,validation_split=0.1)
model.fit(trainX, trainY, nb_epoch=1000, batch_size=32, verbose=2)


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

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
plt.plot(trainPredictPlot[:,0] , color='red', label='Train Phase')  
plt.plot(testPredictPlot[:,0] , color='green', label='Test Phase') 
plt.title('Time Series Prediction') 
plt.xlabel('Time')  
plt.ylabel('Function Value')  
plt.legend() 


plt.figure(2)
#plt.figure(figsize=(18,13))  
plt.plot(dataset[:,1], color='blue', label='Actual Time Series Value')  
plt.plot(trainPredictPlot[:,1] , color='red', label='Train Phase')  
plt.plot(testPredictPlot[:,1] , color='green', label='Test Phase') 
plt.title('Time Series Prediction') 
plt.xlabel('Time')  
plt.ylabel('Function Value')  
plt.legend() 

plt.figure(3)
#plt.figure(figsize=(18,13))  
plt.plot(dataset[:,2], color='blue', label='Actual Time Series Value')  
plt.plot(trainPredictPlot[:,2] , color='red', label='Train Phase')  
plt.plot(testPredictPlot[:,2] , color='green', label='Test Phase') 
plt.title('Time Series Prediction') 
plt.xlabel('Time')  
plt.ylabel('Function Value')  
plt.legend() 

plt.show()









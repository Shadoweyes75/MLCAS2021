
from data_model import *
import numpy as np
import pandas as pd

#just a copy of imports use over multible files
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import logistic
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler

inputs_other = np.load('inputs_others_train.npy')
yield_train = np.load('yield_train.npy')
inputs_weather_train = np.load('inputs_weather_train.npy')
clusterID_genotype = np.load('clusterID_genotype.npy')

'''
This functiion is used to make a model
based on a decsision tree that makes a tree for each of the 
days than uses this to make a prediction over each day
'''
def DecsisionTree(data,day):
    data_set = data
    data_set.head(12)
    y = data_set['yield']
    x = data_set['AvgSur']
    
    y = (y.to_numpy()).reshape(-1, 1)
    x = (x.to_numpy()).reshape(-1, 1)
    #['ADNI','AP','ARH','MDNI','MaxSur','MinSur',]
    test_size = 0.5
    seed = 5
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = seed)
    
    
    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)
    predictions = model.predict(x_train)
    predictions = model.predict(x_test)
    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train)
    
    x_test = pd.DataFrame(x_test)
    y_test = pd.DataFrame(y_test)
      
    test_other = np.load('inputs_others_test.npy')
    inputs_weather_test = np.load('inputs_weather_test.npy')

    sample_two = pull_one_day(inputs_weather_test,day,10337)

    sample_two = pd.DataFrame(sample_two)
    test_other = pd.DataFrame(test_other)

    test_data = group_weather_yeild(sample_two,test_other)
    test_data = test_data['AvgSur']
    test_data = (test_data.to_numpy()).reshape(-1, 1)
    predictions = model.predict(test_data)
    
    return predictions
'''
This function is never use and is just for testing
'''
def model_not_lstm(data):
    
    data_set = data
    
    data_set.head(12)
    
    y = data_set['yield']

    x = data_set['AvgSur']
    #['ADNI','AP','ARH','MDNI','MaxSur','MinSur',]
    x.head()
    y.head()
    test_size = 0.5
    seed = 5
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = seed)
    #model
    model = DecisionTreeClassifier()
    
    model.fit(x_train, y_train)
    
    predictions = model.predict(x_train)
    print(accuracy_score(y_train, predictions))
    predictions = model.predict(x_test)
    print(accuracy_score(y_test, predictions))   
    
    df = x_test.copy()
    df['Actual'] = y_test
    df['Prediction'] = predictions
    df

def lstm_model(data):
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    data_set = data
    data_set.head(12)
    y = data_set['yield'].reshape(-1, 1)
    x = data_set[['ADNI','AP','ARH','MDNI','MaxSur','MinSur','AvgSur']].reshape(-1, 1)
    x.head()
    y.head()
    
    test_size = 0.5
    
    seed = 5
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = seed)
    
    x_train = x_train/255.0
    x_test = x_test/255.0

    model = Sequential()

    model.add(LSTM(128,input_shape=(x_train.shape[1:]),activation ='relu',return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128,activation ='relu'))
    model.add(Dropout(0.2))

    model.add(LSTM(32,activation ='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10,activation='softmax'))

    opt = tf.keras.optimizers.Adam(Lr=1e-3,decay=1e-5)

    model.compile(Loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

    model.fit(x_train,y_train,epochs=2,validation_data=(x_test,y_test))


def main():

    #sample_one = pull_one_day(inputs_weather_train,0,93028)
    #yield_merge = merge_yield_other(inputs_other,yield_train)
    #group_data = group_weather_yeild(sample_one,yield_merge)
    #group_data.head(15)
    #group_data.tail(15)
    #group_data.describe()
    #sample_one.describe()
    #model_not_lstm(group_data)
    #lstm_model(group_data)

    prediction_each_day = []
    for i in range(213):
        sample_one = pull_one_day(inputs_weather_train,i,93028)
        yield_merge = merge_yield_other(inputs_other,yield_train)
        group_data = group_weather_yeild(sample_one,yield_merge)
        prediction_each_day.append(DecsisionTree(group_data,i))
    

    
    print(prediction_each_day[0])
    #predictions = DecsisionTree(group_data)

    #predict.head(15)
    #predict.shape
    #predictions = pd.DataFrame(predictions)
    #predictions.head(15)
    #predictions.shape
    #np.save("predictions",(predictions.to_numpy()))
    tmp = pd.DataFrame(prediction_each_day)
    tmp.shape

    prediction_over_time=[]
    for i in range(10337):
        prediction_over_time.append(tmp[i].mean())

    prediction_over_time = pd.DataFrame(prediction_over_time)
    np.save("prediction_over_time",(prediction_over_time.to_numpy()))
if __name__ == '__main__':
    main()
    



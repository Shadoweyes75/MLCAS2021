
#from data_model import *
import numpy as np
import pandas as pd
#just a copy of imports use over multible files
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
#from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
#from sklearn import datasets, linear_model
#from sklearn.model_selection import cross_val_predict
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import logistic
#from sklearn.linear_model import LinearRegression
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn import svm
#from matplotlib import pyplot
#from sklearn.preprocessing import StandardScaler
import requests




#compress weather
def buildWeather(inputs_weather_train):
    new_weather = np.column_stack((np.repeat(np.arange(93028),214),inputs_weather_train.reshape(93028*214,-1)))
    new_weather = pd.DataFrame(new_weather, columns = ['PR','ADNI','AP','ARH','MDNI','MaxSur','MinSur','AvgSur'])
    return new_weather

#pulls one day out of weather file
def pull_one_day(dataWeather,day,ran):
    dayOne = []
    #dfw = pd.DataFrame(columns = ['AP','ARH','MDNI','MxSur','MnSur','MnSur','AgSur'])
    for i in range(ran):
        dayOne.append(dataWeather[i][day].tolist())
    #dfw.append(pd.DataFrame((dataWeather[i][0].tolist()) ,columns = ['AP','ARH','MDNI','MxSur','MnSur','MnSur','AgSur']))
    dfw = pd.DataFrame(dayOne,columns = ['ADNI','AP','ARH','MDNI','MaxSur','MinSur','AvgSur'])
    return dfw

#compress other data and yeid
def merge_yield_other(inputs_other,yield_train):
    inputs_other_df = pd.DataFrame(inputs_other,columns = ['MG','GID','State','Year','loc'])
    yield_train_df = pd.DataFrame(yield_train,columns = ['yield'])
    inputs_other_df.insert(5,"yield", yield_train_df['yield'])
    return inputs_other_df

#makes the data smaller for testing
def make_data_small(dataFrame,size,random):
    if random:
        return dataFrame.sample(size)
    else:
        return dataFrame.head(size)

#brings weather and yields together
def group_weather_yeild(weather,yields):
    return pd.concat([yields,weather], axis=1)


#url = 'https://data.cyverse.org/dav-anon/iplant/home/nirav/ACIC-2021/Dataset_Competition_Zip_File.zip'
#r = requests.get(url, allow_redirects=True)

#open('Dataset_Competition_Zip_File.zip', 'wb').write(r.content)

#import zipfile
#with zipfile.ZipFile('Dataset_Competition_Zip_File.zip', 'r') as zip_ref:
#    zip_ref.extractall('.')



inputs_other = np.load('/dataset_competition/raining/inputs_others_train.npy')
yield_train = np.load('/dataset_competition/raining/yield_train.npy')
inputs_weather_train = np.load('/dataset_competition/raining/inputs_weather_train.npy')
#clusterID_genotype = np.load('Dataset_Competition\clusterID_genotype.npy')

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
      
    test_other = np.load('/dataset_competition/est_inputs/inputs_others_test.npy')
    inputs_weather_test = np.load('/dataset_competition/est_inputs/inputs_weather_test.npy')

    sample_two = pull_one_day(inputs_weather_test,day,10337)

    sample_two = pd.DataFrame(sample_two)
    test_other = pd.DataFrame(test_other)

    test_data = group_weather_yeild(sample_two,test_other)
    test_data = test_data['AvgSur']
    test_data = (test_data.to_numpy()).reshape(-1, 1)
    predictions = model.predict(test_data)
    
    return predictions
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
    print("Training Model...")
    for i in range(213):
        #print("Training pass-> "+ str(i))
        sample_one = pull_one_day(inputs_weather_train,i,93028)
        yield_merge = merge_yield_other(inputs_other,yield_train)
        group_data = group_weather_yeild(sample_one,yield_merge)
        prediction_each_day.append(DecsisionTree(group_data,i))
    

    print("predicting...")
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
    print("saving results")
    np.save("/save/prediction_over_time",(prediction_over_time.to_numpy()))
    print("completed")
if __name__ == '__main__':
    main()
    



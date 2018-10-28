import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_utils
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score,recall_score,f1_score
from keras.models import Sequential
from keras.layers import Dense
'''
csvname = 'C:\\Users\\93621\\Desktop\\S_dos.csv'
data = pd.read_csv(csvname)
headers = ['SourceIp','DestIp','SourcePort','destPort','Trans_Id','funcCode','Refno','Register_data','Exeption_Code','Time_Stamp']
data.columns = headers
new = [ "Alarm" ]
for i in new:
    data[i] = np.nan
i = 0
data = data.fillna('0')
packets_num = len(data['Alarm'])
IPsource = data['SourceIp'].unique().tolist()
IPdest = data['DestIp'].unique().tolist()
IP = list(set(IPsource + IPdest))
for j in range(0, len(IP)):
    data = data.replace(IP[j],j)
while(i < packets_num - 3):
    if data['Exeption_Code'][i] == 1:
        data['Alarm'][i] = 1
    if len(data['Register_data'][i+1]) > 10:
        data['Register_data'][i + 1] = 123
        if len(data['Register_data'][i+3]) > 10:
            data['Alarm'][i:i+3]  = 1
    i = i + 1
print("Processing is over, start training")
'''
csvname = 'D:\dos_pcap\OCT27_output.csv'
data = pd.read_csv(csvname)
def pre_split(input_data):
    X = input_data[X_Label]
    X = np.array(X,dtype=float)
    y = input_data['Alarm']
    y = np.array(y,dtype=int)
    return X, y
n_classes = 1
#data.to_csv('C:\\Users\\93621\\Desktop\\LogLabeled.csv')
X_Label = [i for i in data.columns.tolist() if i not in 'Alarm']
X, y = pre_split(data)
#X_test, y_test = pre_split(test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

def logistic():
    model = Sequential()
    model.add(Dense(1, input_dim= 11, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(X_test, y_test, batch_size=100, epochs=100)
    scores = model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def NN():
    model = Sequential()
    model.add(Dense(10, input_dim=11,activation='relu'))
    model.add(Dense(5,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    model.fit(X_test, y_test,batch_size=100,epochs=50)
    scores = model.evaluate(X_test,y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
logistic()
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
'''
csvname = 'C:\\Users\\93621\\Desktop\\L_dos.csv'
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
if i == packets_num - 3:
    if len(data['Register_data'][i+1]) > 10:
        data['Register_data'][i + 1] = 123
    if len(data['Register_data'][i+2]) > 10:
        data['Register_data'][i + 1] = 123
print("Processing is over, start training")
data.to_csv('C:\\Users\\93621\\Desktop\\L_LogLabeled.csv')
'''
csvname = 'C:\\Users\\93621\\Desktop\\L_LogLabeled.csv'
data = pd.read_csv(csvname)
testname = 'C:\\Users\\93621\\Desktop\\LogLabeled.csv'
test = pd.read_csv(testname)
X_Label = [i for i in data.columns.tolist() if i not in 'Alarm']
X = data[X_Label]
y = data['Alarm']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X = np.array(X)
X = X.reshape((X.shape[0],1,X.shape[1]))
X_test = np.array(X_test)
test_X = test[X_Label]
test_y = test['Alarm']
test_X = np.array(test_X)
test_X = test_X.reshape((test_X.shape[0],1,test_X.shape[1]))
max_review_length = 10
embedding_vecor_length = 32
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1],X.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X, y, epochs=5, batch_size=64)
scores = model.evaluate(test_X, test_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
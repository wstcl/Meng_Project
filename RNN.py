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

X = [i for i in data.columns.tolist() if i not in 'Alarm']
X = data[X]
y = data['Alarm']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
max_review_length = 10
embedding_vecor_length = 64
model = Sequential()
model.add(Embedding(10000, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(300))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=4, batch_size=64)
scores = model.evaluate(X_test, y_test)
print(scores)
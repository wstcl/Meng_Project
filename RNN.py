import pandas as pd
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split


def data_process(input, output):
    data = pd.read_csv(input)
    headers = ['SourceIp','DestIp','SourcePort','destPort','tcp_stream_Index','Trans_Id','funcCode','Refno','Register_data','Exeption_Code','Time_Stamp','Relative_Time']
    data.columns = headers
    new = ["Alarm" ]
    for i in new:
        data[i] = np.nan
    i = 0
    stream_index = 0
    data = data.fillna('0')
    packets_num = len(data['Alarm'])
    IPsource = data['SourceIp'].unique().tolist()
    IPdest = data['DestIp'].unique().tolist()
    IP = list(set(IPsource + IPdest))
    for j in range(0, len(IP)):
        data = data.replace(IP[j],j)
    while(i < packets_num):
        if data['Exeption_Code'][i] == 1:
            if data['tcp_stream_Index'][i] == stream_index:
                data['Alarm'][index] = 0
            else:
                data['Alarm'][i] = 1
                stream_index = data['tcp_stream_Index'][i]
                index = i

        if data['Trans_Id'][i] == 1:
            data['Alarm'][i] = 1
            if len(data['Register_data'][i]) > 20:
                data['Alarm'][i] = 0
        i = i + 1
    del(data['Register_data'])
    data.to_csv(output, index=False)
    print("Processing is over, start training")



def RNN(csvname):
    data = pd.read_csv(csvname)
    X_Label = [i for i in data.columns.tolist() if i not in 'Alarm']
    X = data[X_Label]
    y = data['Alarm']
    X = np.array(X)
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



    max_review_length = 10
    embedding_vecor_length = 32
    model = Sequential()
    model.add(LSTM(128, input_shape=(X.shape[1],X.shape[2])))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=5, batch_size=64)
    scores = model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


#data_process('D:\dos_pcap\OCT27.csv','D:\dos_pcap\OCT27_output.csv')
RNN('D:\dos_pcap\OCT27_output.csv')
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,f1_score
import h5py

def evaluate(y_pred, y_test):
    pre = precision_score(y_true=y_test,y_pred=y_pred)
    rec = recall_score(y_true=y_test,y_pred=y_pred)
    f1 = f1_score(y_true=y_test,y_pred=y_pred)
    return pre, rec, f1

def data_reshape(dst, label, timesteps):
    input = dst[label]
    Y = dst['Alarm']
    input = np.array(input)
    Y = np.array(Y)
    drop = input.shape[0] % timesteps
    for i in range(drop):
        input = np.delete(input, input.shape[0] - i - 1, axis=0)
        Y = np.delete(Y, Y.shape[0] - i - 1, axis=0)
    X = input.reshape((input.shape[0] // timesteps, timesteps, input.shape[1]))
    y = Y.reshape(Y.shape[0] // timesteps, timesteps)
    return X, y

def data_process(input, output):
    data = pd.read_csv(input)
    headers = ['SourceIp','DestIp','SourcePort','destPort','tcp_stream_Index','Seq_num','Trans_Id','funcCode','Refno','Register_data','Exeption_Code','Time_Stamp','Relative_Time']
    data.columns = headers
    new = ["Alarm" ]
    for i in new:
        data[i] = np.nan
    i = 0
    flag = 0
    data = data.fillna('0')
    packets_num = len(data['Alarm'])
    while(i < packets_num):
        if data['SourceIp'][i] == "192.168.100.11" or data['DestIp'][i] == "192.168.100.11":
            if data['Refno'][i] == 52210:
                data.drop([i],inplace=True)
                i = i + 1
                flag = 1
                continue
            if data['Refno'][i] == 52211:
                data.drop([i],inplace=True)
                i = i + 1
                flag = 0
                continue
            if data['Refno'][i] != 52210 and data['Refno'][i] != 52211 and flag == 1:
                data['Alarm'][i] = 1
        i = i + 1

    IPsource = data['SourceIp'].unique().tolist()
    IPdest = data['DestIp'].unique().tolist()
    IP = list(set(IPsource + IPdest))
    for j in range(0, len(IP)):
        data = data.replace(IP[j], j)
    data.to_csv(output, index=False)
    print("Processing is over, start training")


def RNN(csvname):
    data = pd.read_csv(csvname)
    test = pd.read_csv('D:\\dos_pcap\\nov6_output.csv')
    timestep = 5
    X_Label = [i for i in data.columns.tolist() if i not in 'Alarm']
    X, y = data_reshape(data, X_Label, timestep)
    test_x, test_y = data_reshape(test, X_Label, timestep)

    for kfold in range(1):
        model = Sequential()
        model.add(LSTM(256, return_sequences=True,input_shape=(X.shape[1],X.shape[2])))
        model.add(LSTM(128))
        model.add(Dense(timestep, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        model.fit(X, y, epochs=5, batch_size=128)
        score = model.evaluate(test_x,test_y)
        print(score)
    print("Done")



data_process('D:\\dos_pcap\\nov8L.csv','D:\\dos_pcap\\nov8L_output.csv')
#data_process('D:\\dos_pcap\\nov8mn.csv','D:\\dos_pcap\\nov8mn_output.csv')
RNN('D:\\dos_pcap\\nov8L_output.csv')
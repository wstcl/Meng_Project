import pandas as pd
import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#import h5py
#from keras.models import load_model

IPhash = {'10.0.0.3':297913,
          '10.0.0.4':297914,
          '10.0.0.5':297915,
          '192.168.100.11':5884431}


def f1_score(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0
    # How many selected items are relevant?
    precision = c1 / c2
    # How many relevant items are selected?
    recall = c1 / c3
    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def classreport(y_true, y_pred):
    # Count positive samples.
    c1 = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    c2 = np.sum(np.round(np.clip(y_pred, 0, 1)))
    c3 = np.sum(np.round(np.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0
    # How many selected items are relevant?
    precision = c1 / c2
    # How many relevant items are selected?
    recall = c1 / c3
    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision,recall,f1_score

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
    # IPhash
    data["SourceIp"].replace(IPhash, inplace=True)
    data["DestIp"].replace(IPhash, inplace=True)
    data = data.convert_objects(convert_numeric=True)
    while(i < packets_num):
        if data['SourceIp'][i] == 5884431 or data['DestIp'][i] == 5884431:
            if data['Refno'][i] == 52210:
                index = data['tcp_stream_Index'][i]
                data.drop([i],inplace=True)
                i = i + 1
                flag = 1
                continue
            if data['Refno'][i] == 52211:
                index = data['tcp_stream_Index'][i]
                data.drop([i],inplace=True)
                i = i + 1
                flag = 0
                continue
            if data['Refno'][i] != 52210 and data['Refno'][i] != 52211 and flag == 1:
                if data['tcp_stream_Index'][i] == index:
                    data.drop([i], inplace=True)
                    i = i + 1
                    continue
                data['Alarm'][i] = 1
        i = i + 1

    data = data.convert_objects(convert_numeric=True)
    #Standardization
    feature_mean = []
    feature_std = []
    for i in headers:
        mean = data[i].mean()
        std = data[i].std()
        feature_mean.append(mean)
        feature_std.append(std)
        data[i] = (data[i]-mean)/std
    with open('result.txt',"a") as f:
        f.write("\n")
        f.write('mean and std for: ')
        f.write(input)
        f.write("\n")
        f.write("mean: ")
        for i in feature_mean:
            f.write(str(i))
            f.write(", ")
        f.write('\n')
        f.write("std: ")
        for i in feature_std:
            f.write(str(i))
            f.write(", ")
        f.close()
    print(feature_mean)
    print(feature_std)
    data.to_csv(output, index=False)
    print("Processing is over, start training")


def RNN(csvname,testname):
    data = pd.read_csv(csvname)
    #del(data['tcp_stream_Index'])
    #del(data['Seq_num'])
    test = pd.read_csv(testname)
    #del(test['tcp_stream_Index'])
    #del(test['Seq_num'])
    timestep = 5
    X_Label = [i for i in data.columns.tolist() if i not in 'Alarm']
    X, y = data_reshape(data, X_Label, timestep)
    test_x, test_y = data_reshape(test, X_Label, timestep)
    precision = []
    recall = []
    f1 = []
    for kfold in range(10):
        model = Sequential()
        model.add(LSTM(256,return_sequences=True,input_shape=(X.shape[1],X.shape[2])))
        model.add(LSTM(128))
        model.add(Dense(timestep, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_score])
        print(model.summary())
        model.fit(X, y, epochs=10, batch_size=1000)
        #model.save('lstm.h5')
        y_pre = model.predict(test_x)
        pscore,rsocre,fscore = classreport(y_pred=y_pre,y_true=test_y)

        precision.append(pscore)
        recall.append(rsocre)
        f1.append(fscore)

        score = model.evaluate(test_x,test_y,verbose=1)
        print(score)
    precision = np.array(precision)
    recall = np.array(recall)
    f1 = np.array(f1)
    print("Precision：",np.mean(precision)," +- ", np.std(precision))
    print("recall：",np.mean(recall)," +- ", np.std(recall))
    print("f1：",np.mean(f1)," +- ", np.std(f1))


    print("Done")

data_process('D:\\dos_pcap\\nov8L.csv','D:\\dos_pcap\\nov8L_output.csv')
data_process('D:\\dos_pcap\\nov8mn.csv','D:\\dos_pcap\\nov8mn_output.csv')
RNN('D:\\dos_pcap\\nov8L_output.csv','D:\\dos_pcap\\nov8mn_output.csv')


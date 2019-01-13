import pandas as pd
import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import h5py
from keras.models import load_model
import matplotlib.pyplot as plt
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

def data_reshape(dst, label, timesteps,feature_mean=[], feature_std=[]):
    if feature_mean != []:
        mean = feature_mean
        std = feature_std
        index = 0
        for i in label:
            mean = feature_mean[index]
            std = feature_std[index]
            dst[i] = (dst[i] - mean) / std
            index = index + 1
        print("feature_mean:", feature_mean)
        print("feature_std:", feature_std)
    if feature_mean == []:
        #Standardization
        feature_mean = []
        feature_std = []
        for i in label:
            mean = dst[i].mean()
            std = dst[i].std()
            feature_mean.append(mean)
            feature_std.append(std)
            dst[i] = (dst[i] - mean) / std
        print("feature_mean:", feature_mean)
        print("feature_std:", feature_std)

    #reshape
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
    return X, y,feature_mean,feature_std

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

    data.to_csv(output, index=False)
    print("Processing is over, start training")


def RNN(csvname,testname):
    data = pd.read_csv(csvname)
    del(data['tcp_stream_Index'])
    #del(data['Seq_num'])
    test = pd.read_csv(testname)
    del(test['tcp_stream_Index'])
    #del(test['Seq_num'])
    timestep = 5
    X_Label = [i for i in data.columns.tolist() if i not in 'Alarm']
    X, y,features_mean,features_std = data_reshape(data, X_Label, timestep)
    test_x, test_y,features_mean,features_std = data_reshape(test, X_Label, timestep,features_mean,features_std)
    precision = []
    recall = []
    f1 = []
    for kfold in range(1):
        model = Sequential()
        model.add(LSTM(256,return_sequences=True,input_shape=(X.shape[1],X.shape[2])))
        model.add(LSTM(128))
        model.add(Dense(timestep, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        print(model.summary())
        history = model.fit(X, y, epochs=40, batch_size=1000,validation_data=[test_x,test_y])
        model.save('lstm.h5')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['J$_{train}$', 'J$_{cv}$'], loc='upper right')
        plt.show()
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

def Performance_evaluation(onlinecsv,output):
    data = pd.read_csv(onlinecsv)
    data = np.array(data)
    y_true = data.copy()
    start_index = np.argwhere(data[:,7]==52210)
    end_index = np.argwhere(data[:,7]==52211)
    start_index = np.concatenate(start_index)
    end_index = np.concatenate(end_index)
    Flag_Diff = len(start_index)-len(end_index)
    if Flag_Diff==0:
        for i in range(len(start_index)):
            A_index = np.argwhere(data[start_index[i]:end_index[i],0:2]==5884431) + start_index[i]
            y_true[A_index[:,0],12]=1
    elif Flag_Diff == 1:
        for i in range(len(start_index)-1):
            A_index = np.argwhere(data[start_index[i]:end_index[i],0:2]==5884431) + start_index[i]
            y_true[A_index[:,0],12]=1
        A_index = np.argwhere(data[start_index[-1]:len(data),0:2]==5884431) + start_index[i]
        y_true[A_index[:,0],12]=1
    else:
         raise ValueError('Difference between start and end flag is greater than one.')
    start_flags = np.append(start_index,start_index+1)
    end_flags= np.append(end_index,end_index+1)
    flags=np.append(start_flags,end_flags)
    y_true = np.delete(y_true,flags,axis = 0)
    data = np.delete(data,flags,axis=0)
    truelabel = y_true[:,12]
    prelabel = data[:,12]
    print(classreport(y_pred=prelabel,y_true=truelabel))
    np.savetxt(output,y_true,delimiter=',')

#data_process('D:\\dos_pcap\\Dec2.csv','D:\\dos_pcap\\Dec2_output.csv')
#data_process('D:\\dos_pcap\\Dec2_4.csv','D:\\dos_pcap\\Dec2_4_output.csv')
#RNN('D:\\dos_pcap\\Dec2_output.csv','D:\\dos_pcap\\Dec2_4_output.csv')
Performance_evaluation('D:\dos_pcap\dos_csv\Log.csv','D:\dos_pcap\dos_csv\Label.csv')

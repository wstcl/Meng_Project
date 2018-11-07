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

def Is_stream_change(new,old):
    if new == old:
        return 0
    else:
        return 1

def data_process(input, output):
    data = pd.read_csv(input)
    headers = ['SourceIp','DestIp','SourcePort','destPort','tcp_stream_Index','Seq_num','Trans_Id','funcCode','Refno','Register_data','Exeption_Code','Time_Stamp','Relative_Time']
    data.columns = headers
    new = ["Alarm" ]
    for i in new:
        data[i] = np.nan
    i = 0
    stream_index = 0
    data = data.fillna('0')
    packets_num = len(data['Alarm'])
    ukp_index = []
    scan_index = 0
    while(i < packets_num):
        if data['Exeption_Code'][i] == 1:
            if data['tcp_stream_Index'][i] == stream_index:
                data['Alarm'][index] = 0
            else:
                data['Alarm'][i] = 1
                stream_index = data['tcp_stream_Index'][i]
                index = i

        if data['funcCode'][i] == 3:
            if data['SourceIp'][i] == "192.168.100.11" or data['DestIp'][i] == "192.168.100.11":    # 192.168.100.11 could be replaced by a out net e.g. if data not in [inner_net]
                if data['Trans_Id'][i] == 1:
                    if data['tcp_stream_Index'][i] != scan_index:
                        scan_index = data['tcp_stream_Index'][i]
                        data['Alarm'][i] = 1
                        ukp_index.append(i)
                    if data['tcp_stream_Index'][i] == scan_index and data['DestIp'][i] == '192.168.100.11':
                        data['Alarm'][i] = 1
                        ukp_index.append(i)
                if data['Trans_Id'][i] == 2:
                    if len(ukp_index) > 1:
                        data['Alarm'][ukp_index[-1],ukp_index[-2]] = 0
                        ukp_index = []
                    if len(ukp_index) == 1:
                        data['Alarm'][ukp_index[-1]] = 0
                        ukp_index = []


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
    #test = pd.read_csv('D:\dos_pcap\OCT30MN_output.csv')
    X_Label = [i for i in data.columns.tolist() if i not in 'Alarm']
    X = data[X_Label]
    y = data['Alarm']
    X = np.array(X)
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    P_score = []
    R_score = []
    F_score = []
    '''test_x = test[X_Label]
    test_y = test['Alarm']
    test_x = np.array(test_x)
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))'''

    for kfold in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
        model = Sequential()
        model.add(LSTM(256, return_sequences=True,input_shape=(X_train.shape[1],X_train.shape[2])))
        model.add(LSTM(128))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        model.fit(X_train, y_train, epochs=15, batch_size=500)
        y_pre = model.predict_classes(X_test)
        model.save('D:\SCADA_PROJECT\my_model.h5')
        precision, recall, f1 = evaluate(y_pre,y_test)
        P_score.append(precision)
        R_score.append(recall)
        F_score.append(f1)
    P_score = np.array(P_score)
    R_score = np.array(R_score)
    F_score = np.array(F_score)
    print("Precision: ", np.mean(P_score),"+-" , np.std(P_score))
    print("Recall: ", np.mean(R_score), "+-", np.std(R_score))
    print("F1_socre: ", np.mean(F_score), "+-", np.std(F_score))

#data_process('D:\\dos_pcap\\nov6.csv','D:\\dos_pcap\\nov6_output.csv')
RNN('D:\\dos_pcap\\OCT30MN_output.csv')
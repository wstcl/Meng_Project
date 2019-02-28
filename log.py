import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix
import matplotlib.pyplot as plt
def evaluate(y_pred, y_test):
    pre = precision_score(y_true=y_test,y_pred=y_pred)
    rec = recall_score(y_true=y_test,y_pred=y_pred)
    f1 = f1_score(y_true=y_test,y_pred=y_pred)
    return pre, rec, f1

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self,batch,logs={}):
        self.losses.append(logs.get('loss'))
        self.val_loss = self.model.evaluate(self.validation_data[0],self.validation_data[1])
        self.val_losses.append(self.val_loss[0])

def pre_split(input_data, Label):
    X = input_data[Label]
    #del(X['tcp_stream_Index'])
    feature_mean = []
    feature_std = []
    for i in Label:
        mean = X[i].mean()
        std = X[i].std()
        feature_mean.append(mean)
        feature_std.append(std)
        X[i] = (X[i] - mean) / std
    print("feature_mean:", feature_mean)
    print("feature_std:", feature_std)
    X = np.array(X,dtype=float)
    y = input_data['Alarm']
    y = np.array(y,dtype=int)
    return X, y

def logistic(csvname):
    data = pd.read_csv(csvname)
    headers = ['SourceIp','DestIp','SourcePort','destPort','Seq_num','Trans_Id','funcCode','Refno','Register_data','Exeption_Code','Time_Stamp','Relative_Time','Alarm']
    data.columns = headers
    X_Label = [i for i in data.columns.tolist() if i not in 'Alarm']
    X, y = pre_split(data, X_Label)
    Train_Error = []
    Test_Error = []
    for kfold in range(1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        num_seq = X_train.shape[0]
        num_sample = [1, 10, 50, 100, 500, 1000, 5000, 10000, 30000, 50000, 100000, num_seq]
        for sample_size in num_sample:
            model = Sequential()
            model.reset_states()
            model.add(Dense(1,input_dim= 12,activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
            model.fit(X_train[0:sample_size, :], y_train[0:sample_size], epochs=30, shuffle=False)
            Train_loss = model.evaluate(X_train[0:sample_size, :], y_train[0:sample_size])
            Validation_loss = model.evaluate(X_test, y_test)
            Train_Error.append(Train_loss[0])
            Test_Error.append(Validation_loss[0])
        plt.figure()
        plt.plot(num_sample, Train_Error)
        plt.plot(num_sample, Test_Error)
        plt.xlabel('Size of training')
        plt.ylabel('Loss')
        plt.legend(['J$_{train}$', 'J$_{cv}$'], loc='upper right')
        plt.savefig('C:\\Users\\93621\\Desktop\\SCADA plot\\Logistic_learning.png')


def NN(csvname):
    data = pd.read_csv(csvname)
    headers = ['SourceIp','DestIp','SourcePort','destPort','Seq_num','Trans_Id','funcCode','Refno','Register_data','Exeption_Code','Time_Stamp','Relative_Time','Alarm']
    data.columns = headers
    X_Label = [i for i in data.columns.tolist() if i not in 'Alarm']
    X, y = pre_split(data, X_Label)
    Train_Error = []
    Test_Error = []
    for kfold in range(1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        num_seq = X_train.shape[0]
        num_sample = [1, 10, 100, 1000, 5000, 10000, 50000, 100000, num_seq]
        for sample_size in num_sample:
            model = Sequential()
            model.reset_states()
            model.add(Dense(256, input_dim=12,activation='tanh'))
            model.add(Dense(128,activation='tanh'))
            model.add(Dense(1,activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
            model.fit(X_train[0:sample_size, :], y_train[0:sample_size], epochs=30, shuffle=False)
            Train_loss = model.evaluate(X_train[0:sample_size, :], y_train[0:sample_size])
            Validation_loss = model.evaluate(X_test, y_test)
            Train_Error.append(Train_loss[0])
            Test_Error.append(Validation_loss[0])
        plt.figure()
        plt.plot(num_sample, Train_Error)
        plt.plot(num_sample, Test_Error)
        plt.xlabel('Size of training')
        plt.ylabel('Loss')
        plt.legend(['J$_{train}$', 'J$_{cv}$'], loc='upper right')
        plt.savefig('C:\\Users\\93621\\Desktop\\SCADA plot\\NN_learning.png')
logistic('D:\\dos_pcap\\Label_Jan27b.csv')
NN('D:\\dos_pcap\\Label_Jan27b.csv')

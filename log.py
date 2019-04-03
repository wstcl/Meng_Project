import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
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
    es = EarlyStopping(monitor='loss',min_delta = 0.00005, patience = 5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    num_seq = X_train.shape[0]
    num_sample = [10, 100, 500, 1000, 5000, 10000, 30000, 50000, 80000, 100000, 150000, 200000, num_seq]
    
    for sample_size in num_sample:
        print(sample_size,file=open("10fold/Log_10_fold_train.txt","a"))
        print("Precision",',',"Recall",',',"F1",',',"Accuray",',',"loss",file=open("10fold/Log_10_fold_train.txt","a"))
        print(sample_size,file=open("10fold/Log_10_fold_test.txt","a"))
        print("Precision",',',"Recall",',',"F1",',',"Accuray",',',"loss",file=open("10fold/Log_10_fold_test.txt","a"))
        
        for kfold in range(10):
            model = Sequential()
            model.reset_states()
            model.add(Dense(1,input_dim= 12,activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
            model.fit(X_train[0:sample_size, :], y_train[0:sample_size], batch_size = 256, epochs=10000, shuffle=True,callbacks=[es])
            y_pre_train = model.predict_classes(X_train[0:sample_size, :])
            y_pre_test = model.predict_classes(X_test)
            y_pre_train = y_pre_train.reshape(sample_size)
            ftrain = f1_score(y_true=y_train[0:sample_size],y_pred=y_pre_train)
            ptrain = precision_score(y_true=y_train[0:sample_size],y_pred=y_pre_train)
            rtrain = recall_score(y_true=y_train[0:sample_size],y_pred=y_pre_train)
            
            ftest = f1_score(y_true=y_test,y_pred=y_pre_test)
            ptest = precision_score(y_true=y_test,y_pred=y_pre_test)
            rtest = recall_score(y_true=y_test,y_pred=y_pre_test)

            atrain = model.evaluate(X_train[0:sample_size, :], y_train[0:sample_size])
            atest = model.evaluate(X_test,y_test)
            acc_train = atrain[1]
            acc_test = atest[1]
            loss_train = atrain[0]
            loss_test = atest[0]
            print(ptrain,',',rtrain,',',ftrain,',',acc_train,',',loss_train,file=open("10fold/Log_10_fold_train.txt","a"))
            print(ptest,',',rtest,',',ftest,',',acc_test,',',loss_test,file=open("10fold/Log_10_fold_test.txt","a"))
            


def NN(csvname):
    data = pd.read_csv(csvname)
    headers = ['SourceIp','DestIp','SourcePort','destPort','Seq_num','Trans_Id','funcCode','Refno','Register_data','Exeption_Code','Time_Stamp','Relative_Time','Alarm']
    data.columns = headers
    X_Label = [i for i in data.columns.tolist() if i not in 'Alarm']
    X, y = pre_split(data, X_Label)
    Train_Error = []
    Test_Error = []
    es = EarlyStopping(monitor='loss',min_delta = 0.00005, patience = 5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    num_seq = X_train.shape[0]
    num_sample = [10, 100, 500, 1000, 5000, 10000, 30000, 50000, 80000, 100000, 150000, 200000, num_seq]
    
    for sample_size in num_sample:
        print(sample_size,file=open("10fold/NN_10_fold_train.txt","a"))
        print("Precision",',',"Recall",',',"F1",',',"Accuray",',',"loss",file=open("10fold/NN_10_fold_train.txt","a"))
        print(sample_size,file=open("10fold/NN_10_fold_test.txt","a"))
        print("Precision",',',"Recall",',',"F1",',',"Accuray",',',"loss",file=open("10fold/NN_10_fold_test.txt","a"))
        
        for kfold in range(10):
            model = Sequential()
            model.reset_states()
            model.add(Dense(256, input_dim=12,activation='tanh'))
            model.add(Dense(128,activation='tanh'))
            model.add(Dense(1,input_dim= 12,activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
            model.fit(X_train[0:sample_size, :], y_train[0:sample_size], batch_size=256, epochs=10000, shuffle=True,callbacks=[es])
            y_pre_train = model.predict_classes(X_train[0:sample_size, :])
            y_pre_test = model.predict_classes(X_test)
            y_pre_train = y_pre_train.reshape(sample_size)

            ftrain = f1_score(y_true=y_train[0:sample_size],y_pred=y_pre_train)
            ptrain = precision_score(y_true=y_train[0:sample_size],y_pred=y_pre_train)
            rtrain = recall_score(y_true=y_train[0:sample_size],y_pred=y_pre_train)
            
            ftest = f1_score(y_true=y_test,y_pred=y_pre_test)
            ptest = precision_score(y_true=y_test,y_pred=y_pre_test)
            rtest = recall_score(y_true=y_test,y_pred=y_pre_test)

            atrain = model.evaluate(X_train[0:sample_size, :], y_train[0:sample_size])
            atest = model.evaluate(X_test,y_test)
            acc_train = atrain[1]
            acc_test = atest[1]
            loss_train = atrain[0]
            loss_test = atest[0]
    

logistic('Label_Jan27b.csv')
NN('Label_Jan27b.csv')

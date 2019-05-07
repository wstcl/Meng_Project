import pandas as pd
import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
import keras
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import random
import time
import os
import h5py
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
IPhash = {'10.0.0.3':297913,
          '10.0.0.4':297914,
          '10.0.0.5':297915,
          '192.168.100.11':5884431}


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



def data_reshape_multilabel(dst, label, timesteps,feature_mean=[], feature_std=[],model=1):
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
            if std == 0:
                std=1
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
    Y = to_categorical(Y)
    print(Y)
    drop = input.shape[0] % timesteps
    for i in range(drop):
        input = np.delete(input, input.shape[0] - i - 1, axis=0)
        Y = np.delete(Y, Y.shape[0] - i - 1, axis=0)
    X = input.reshape((input.shape[0]//timesteps, timesteps, input.shape[1]))
    if model == 0:    #many-one
        y = Y.reshape(Y.shape[0] // timesteps, timesteps*3)
    if model == 1:    #many-many
        y = Y.reshape(Y.shape[0]//timesteps,timesteps,2)
    return X, y,feature_mean,feature_std

def pre_split(X,y,test_size):

    print(X.shape)
    num_packets = X.shape[0]

    y = np.array(y,dtype=int)
    #y = to_categorical(y)
    test_indx =np.array(random.sample(range(num_packets),int(test_size*num_packets)))

    train_indx = np.setdiff1d(np.array(range(num_packets)),test_indx)
    X_train = X[train_indx]
    y_train = y[train_indx]
    X_test = X[test_indx]
    y_test = y[test_indx]
    return X_train, y_train, X_test,y_test,train_indx,test_indx

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
            if std == 0:
                std=1
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
    X = input.reshape((input.shape[0]//timesteps, timesteps, input.shape[1]))
    y = Y.reshape(Y.shape[0] // timesteps, timesteps)
    return X, y,feature_mean,feature_std

def miso_prepare(data,timesteps):
    data = np.array(data)
    n_p = data.shape[0]
    X_raw = data[:,:-1]
    X_raw = preprocessing.scale(X_raw)
    X = np.zeros((n_p-timesteps,timesteps,data.shape[1]-1))
    y = np.zeros((n_p-timesteps,timesteps)) #need to be rescheduled for multi_label
    #print(X.shape)
    y_raw = data[:,-1] #reschedule for multi-label
    for i in range(timesteps,n_p):
        X[i-timesteps]=X_raw[i-timesteps:i,:]
        y[i-timesteps]=y_raw[i-timesteps:i]
    print(X.shape)
    print(y.shape)
    return X,y


    




def RNN(csvname):
    data = pd.read_csv(csvname)
    data = data.dropna()
    #headers = ['SourceIp', 'DestIp', 'SourcePort', 'destPort', 'Seq_num', 'Trans_Id', 'funcCode', 'Refno',
               #'Register_data', 'Exeption_Code', 'Time_Stamp', 'Relative_Time', 'Alarm']

    #headers = ['SourceIp','DestIp','SourcePort','destPort','Seq_num','Trans_Id','funcCode','Refno','Register_data','Exeption_Code','Time_Stamp','Relative_Time','HH','LL','H','L','speed','t1','t2','Alarm']
    #data.columns = headers
    print("please add comment:")
    comment = input()
    tm = time.ctime() +'_'+comment+ '/'
    path = '/opt/jungao/RNN_results/'
    os.mkdir(path + tm)
    os.mkdir(path + tm + 'model/')
    train = path + tm + "train.csv"
    test = path + tm + "test.csv"

    indices_test = path + tm + "indices_test.csv"
    
    model_path = path + tm + 'model/'
    #del(data['eth_src'])
    #del(data['eth_dst'])

    timestep = 10
    #X_Label = [i for i in data.columns.tolist() if i not in 'Alarm']
    #X, y,features_mean,features_std = data_reshape(data, X_Label, timestep)
    X,y = miso_prepare(data,timestep)
    X_train, y_train, X_test, y_test, train_indx, test_indx=pre_split(X,y,test_size=0.3)
    num_seq = X_train.shape[0]
    #num_sample = [22,50,100,300,500,800,1000,2000,3000,5000,8000,10000,30000,50000,80000,100000,130000,num_seq]
    num_sample = [80000,130000,num_seq]
    neurons = [2,1,4,2,8,4,16,8,32,16,64,32,128,64,256,128,512,256]
    for sample_size in num_sample:
        es = EarlyStopping(monitor='loss',min_delta=1e-5, patience = 1)
        print(sample_size,file=open(train,"a"))
        print("Precision",',',"Recall",',',"F1",',',"Accuray",',',"loss",file=open(train,"a"))
        print(sample_size,file=open(test,"a"))
        print("Precision",',',"Recall",',',"F1",',',"Accuray",',',"loss",file=open(test,"a"))
        
        
        for kfold in range(10):
            model = Sequential()
            
            mc = ModelCheckpoint(model_path+"best.h5",monitor='loss',save_best_only=True,verbose=1,mode='min')
            model.reset_states()
            indx = np.array(random.sample(range(X_train.shape[0]), sample_size))
            model.add(LSTM(64,return_sequences=True,input_shape=(X.shape[1],X.shape[2])))
            model.add(LSTM(32))

            model.add(Dense(timestep, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
            print(model.summary())
            model.fit(X_train[indx], y_train[indx],epochs=5000,batch_size =1000,shuffle = True,callbacks=[es,mc])
            model = load_model(model_path+"best.h5")
            y_pre = model.predict(X_train[indx])
            y_pre = y_pre.reshape((y_pre.shape[0]*timestep,1))
            y_pre[y_pre>=0.5]=1
            y_pre[y_pre<0.5]=0
            #y_pre = np.argmax(y_pre,axis=1)
            y_true = y_train[indx].reshape((y_train[indx].shape[0]*timestep,1))
            #y_true = np.argmax(y_true,axis=1)
            #y_true = y_train[indx]
            y_pre_test = model.predict(X_test)
            y_pre_test = y_pre_test.reshape((y_pre_test.shape[0] * timestep, 1))
            y_pre_test[y_pre_test>=0.5]=1
            y_pre_test[y_pre_test<0.5]=0
            #y_pre_test = np.argmax(y_pre_test, axis=1)
            y_test_true = y_test.reshape((y_test.shape[0]*timestep,1))
            #y_test_true = np.argmax(y_test_true,axis=1)
            #y_test_true = y_test
            ptrain, rtrain, ftrain = evaluate(y_pre, y_true)
            ptest, rtest, ftest = evaluate(y_pre_test, y_test_true)

            atrain = model.evaluate(X_train[indx],y_train[indx],steps=1)
            atest = model.evaluate(X_test,y_test,steps=1)
            acc_train = atrain[1]
            acc_test = atest[1]
            loss_train = atrain[0]
            loss_test = atest[0]
            print(ptrain, ',', rtrain, ',', ftrain, ',', acc_train, ',', loss_train, file=open(train, "a"))
            print(ptest, ',', rtest, ',', ftest, ',', acc_test, ',', loss_test, file=open(test, "a"))
            if sample_size==num_seq:
                labels = np.zeros((2, y_pre_test.shape[0]))
                labels[0, :] = np.transpose(y_test_true)
                labels[1, :] = np.transpose(y_pre_test)
                f = open(indices_test,'a')
                np.savetxt(f,labels,delimiter=',')
                f.close()


    print("Done")
def Performance_evaluation(onlinecsv,output):
    exter_eth = int('000c29d7eaec',16)
    label_index = 14
    data = pd.read_csv(onlinecsv)
    data = np.array(data)
    y_true = data.copy()
    y_true[:,label_index] = 0
    start_index = np.argwhere(data[:,7]==52210)
    end_index = np.argwhere(data[:,7]==52211)
    start_index = np.concatenate(start_index)
    end_index = np.concatenate(end_index)
    Flag_Diff = len(start_index)-len(end_index)
    exter_index = np.argwhere(data[:,12:label_index]==exter_eth)
    mtim_index = np.argwhere(data[exter_index[:,0],0:2]==297913)
    mtim_index = exter_index[mtim_index[:,0],0]
    #mtim_index = np.concatenate(mtim_index)
    y_true[mtim_index,label_index] = 1
    if Flag_Diff==0:
        for i in range(len(start_index)):
            A_index = np.argwhere(data[start_index[i]:end_index[i],0:2]==5884431)+start_index[i]
            y_true[A_index[:,0],label_index]=1
    elif Flag_Diff == 1:
        for i in range(len(start_index)-1):
            A_index = np.argwhere(data[start_index[i]:end_index[i],0:2]==5884431)+start_index[i]
            y_true[A_index[:,0],label_index]=1
        A_index = np.argwhere(data[start_index[-1]:len(data),0:2]==5884431)+start_index[i]
        y_true[A_index[:,0],label_index]=1
    else:
         raise ValueError('Difference between start and end flag is greater than one.')
    flags=np.append(start_index,end_index)
    y_true = np.delete(y_true,flags,axis = 0)
    data = np.delete(data,flags,axis=0)
    print(classreport(y_true[:,label_index],data[:,label_index]))
    np.savetxt(output,y_true,delimiter=',')

def Performance_evaluation_multilabel(onlinecsv,output):
    exter_eth = int('000c29d7eaec',16)
    label_index = 14
    data = pd.read_csv(onlinecsv)
    data = np.array(data)
    y_true = data.copy()
    y_true[:,label_index] = 0
    start_index = np.argwhere(data[:,7]==52210)
    end_index = np.argwhere(data[:,7]==52211)
    start_index = np.concatenate(start_index)
    end_index = np.concatenate(end_index)
    Flag_Diff = len(start_index)-len(end_index)
    exter_index = np.argwhere(data[:,12:label_index]==exter_eth)
    mtim_index = np.argwhere(data[exter_index[:,0],0:2]==297913)
    mtim_index = exter_index[mtim_index[:,0],0]
    #mtim_index = np.concatenate(mtim_index)
    y_true[mtim_index,label_index] = 2
    if Flag_Diff==0:
        for i in range(len(start_index)):
            A_index = np.argwhere(data[start_index[i]:end_index[i],0:2]==5884431)+start_index[i]
            y_true[A_index[:,0],label_index]=1
    elif Flag_Diff == 1:
        for i in range(len(start_index)-1):
            A_index = np.argwhere(data[start_index[i]:end_index[i],0:2]==5884431)+start_index[i]
            y_true[A_index[:,0],label_index]=1
        A_index = np.argwhere(data[start_index[-1]:len(data),0:2]==5884431)+start_index[i]
        y_true[A_index[:,0],label_index]=1
    else:
         raise ValueError('Difference between start and end flag is greater than one.')
    flags=np.append(start_index,end_index)
    y_true = np.delete(y_true,flags,axis = 0)
    data = np.delete(data,flags,axis=0)
    print(classreport(y_true[:,label_index],data[:,label_index]))
    np.savetxt(output,y_true,delimiter=',')


#data_process('D:\\dos_pcap\\Dec2_4.csv','D:\\dos_pcap\\Dec2_4_output.csv')
#Performance_evaluation('mtim.csv','Label_Mtim.csv')
#RNN('pcap file/label_AN_3.csv')
RNN('pcap file/label_mitm.csv')
#Performance_evaluation_multilabel('mtim.csv','Mlabel_Mtim.csv')

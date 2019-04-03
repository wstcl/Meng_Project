import pandas as pd
import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#import h5py
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
IPhash = {'10.0.0.3':297913,
          '10.0.0.4':297914,
          '10.0.0.5':297915,
          '192.168.100.11':5884431}

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self,batch,logs={}):
        self.losses.append(logs.get('loss'))
        self.val_loss = self.model.evaluate(self.validation_data[0],self.validation_data[1])
        self.val_losses.append(self.val_loss[0])

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

def data_reshape_multilabel(dst, label, timesteps,feature_mean=[], feature_std=[],model=0):
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
        y = Y.reshape(Y.shape[0]//timesteps,timestep,3)
    return X, y,feature_mean,feature_std


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
    X = input.reshape((input.shape[0]//timesteps, timesteps, input.shape[1]))
    y = Y.reshape(Y.shape[0] // timesteps, timesteps)
    return X, y,feature_mean,feature_std

def RNN(csvname):
    data = pd.read_csv(csvname)
    headers = ['SourceIp','DestIp','SourcePort','destPort','Seq_num','Trans_Id','funcCode','Refno','Register_data','Exeption_Code','Time_Stamp','Relative_Time','Alarm']
    data.columns = headers
    #del(data['eth_src'])
    #del(data['eth_dst'])
    #test = pd.read_csv(testname)
    #test.columns = headers
    #del(test['eth_src'])
    #del(test['eth_dst'])
    timestep = 10
    X_Label = [i for i in data.columns.tolist() if i not in 'Alarm']
    X, y,features_mean,features_std = data_reshape(data, X_Label, timestep)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
    #test_x, test_y,features_mean,features_std = data_reshape(test, X_Label, timestep,features_mean,features_std)
    num_seq = X.shape[0]
    num_sample = [10,100,500,1000,3000,6000,10000,13000,16000,20000,num_seq]
    Train_Error = []
    Test_Error = []
    neurons = [2,1,4,2,8,4,16,8,32,16,64,32,128,64,256,128,512,256]
    es = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00009, patience=5)
    for sample_size in num_sample:
        print(sample_size,file=open("10fold/LSTM_10_fold_train.txt","a"))
        print("Precision",',',"Recall",',',"F1",',',"Accuray",',',"loss",file=open("10fold/LSTM_10_fold_train.txt","a"))
        print(sample_size,file=open("10fold/LSTM_10_fold_test.txt","a"))
        print("Precision",',',"Recall",',',"F1",',',"Accuray",',',"loss",file=open("10fold/LSTM_10_fold_test.txt","a"))
        
        for kfold in range(10):
            model = Sequential()
            model.add(LSTM(256,return_sequences=True,input_shape=(X.shape[1],X.shape[2])))
            model.add(LSTM(128))
            model.add(Dense(timestep, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
            print(model.summary())
            model.fit(X_train[0:sample_size,:,:], y_train[0:sample_size,:], epochs=5000,shuffle = True,callbacks=[es])
            y_pre_train = model.predict(X_train[0:sample_size,:,:])
            y_pre_test = model.predict(X_test)
            y_true_train = y_train[0:sample_size,:]
            ptrain,rtrain,ftrain = classreport(y_pred=y_pre_train,y_true=y_true_train)
            ptest,rtest,ftest = classreport(y_pred=y_pre_test,y_true=y_test)
            atest = model.evaluate(X_test,y_test)
            atrain = model.evaluate(X_train[0:sample_size,:,:], y_train[0:sample_size,:])
            acc_train = atrain[1]
            acc_test = atest[1]
            loss_train = atrain[0]
            loss_test = atest[0]
            print(ptrain,',',rtrain,',',ftrain,',',acc_train,',',loss_train,file=open("10fold/LSTM_10_fold_train.txt","a"))
            print(ptest,',',rtest,',',ftest,',',acc_test,',',loss_test,file=open("10fold/LSTM_10_fold_test.txt","a"))
             #Train_Error.append(ftrain)
             #Test_Error.append(ftest)
        #print(Train_Error,file=open("RNN_Neuron_Train.txt","a"))
        #print(Test_Error,file=open("RNN_Neuron_Test.txt","a"))
        '''
        plt.figure()
        plt.plot(np.array([2,4,8,16,32,64,128,256,512,1024]),Train_Error)
        plt.plot(np.array([2,4,8,16,32,64,128,256,512,1024]), Test_Error)
        plt.xlabel('Neurons')
        plt.ylabel('F1_score')
        plt.legend(['J$_{train}$','J$_{cv}$'],loc='upper right')
        plt.savefig('RNN_neurons_f1.png')'''

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
RNN('Label_Jan27b.csv')
#Performance_evaluation_multilabel('mtim.csv','Mlabel_Mtim.csv')

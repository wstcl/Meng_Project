import pandas as pd
import sys
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn import preprocessing
from keras import initializers,optimizers,losses
import random
import time
import os
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

def pre_split(input_data, Label,test_size):
    X = input_data[Label]
    X = np.array(X)
    X = preprocessing.scale(X)
    print(X.shape)
    num_packets = X.shape[0]
    y = input_data['Alarm']
    y = np.array(y,dtype=int)
    #y = to_categorical(y)
    test_indx =np.array(random.sample(range(num_packets),int(test_size*num_packets)))
    #np.savetxt("ori_label.csv",np.argmax(y[test_indx],axis=1))
    train_indx = np.setdiff1d(np.array(range(num_packets)),test_indx)
    X_train = X[train_indx]
    y_train = y[train_indx]
    X_test = X[test_indx]
    y_test = y[test_indx]
    return X_train, y_train, X_test,y_test,train_indx,test_indx

def logistic(csvname):
    data = pd.read_csv(csvname)
    #headers = ['SourceIp','DestIp','SourcePort','destPort','Seq_num','Trans_Id','funcCode','Refno','Register_data','Exeption_Code','Time_Stamp','Relative_Time','eth_src','eth_dst','Alarm']
    headers = ['SourceIp','DestIp','SourcePort','destPort','Seq_num','Trans_Id','funcCode','Refno','Register_data','Exeption_Code','Time_Stamp','Relative_Time','eth_src','eth_dst','Alarm']

    data.columns = headers
    del(data['eth_src'])
    del(data['eth_dst'])
    X_Label = [i for i in data.columns.tolist() if i not in 'Alarm']
    X, y = pre_split(data, X_Label)
    Train_Error = []
    Test_Error = []
    es = EarlyStopping(monitor='loss',min_delta = 0.00005, patience = 5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=400)
    num_seq = X_train.shape[0]
    #num_sample = [10, 100, 500, 1000, 5000, 10000, 30000, 50000, 80000, 100000, 150000, 200000, num_seq]
    num_sample = [num_seq]
    for sample_size in num_sample:
        #print(sample_size,file=open("10fold/Log_10_fold_train.txt","a"))
        #print("Precision",',',"Recall",',',"F1",',',"Accuray",',',"loss",file=open("10fold/Log_10_fold_train.txt","a"))
        #print(sample_size,file=open("10fold/Log_10_fold_test.txt","a"))
        #print("Precision",',',"Recall",',',"F1",',',"Accuray",',',"loss",file=open("10fold/Log_10_fold_test.txt","a"))
        
        for kfold in range(1):
            model = Sequential()
            model.reset_states()
            model.add(Dense(y.shape[1],input_dim= 12,activation='sigmoid'))
            model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['categorical_accuracy'])
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
            #print(ptrain,',',rtrain,',',ftrain,',',acc_train,',',loss_train,file=open("10fold/Log_10_fold_train.txt","a"))
            #print(ptest,',',rtest,',',ftest,',',acc_test,',',loss_test,file=open("10fold/Log_10_fold_test.txt","a"))
            



def logistic_nondos(csvname):
    if len(sys.argv)==1:
        cm = ''
    else:
        cm = sys.argv[1]
    data = pd.read_csv(csvname)
    #headers = ['SourceIp','DestIp','SourcePort','destPort','Seq_num','Trans_Id','funcCode','Refno','Register_data','Exeption_Code','Time_Stamp','Relative_Time','Alarm']
    headers = ['SourceIp','DestIp','SourcePort','destPort','Seq_num','Trans_Id','funcCode','Refno','Register_data','Exeption_Code','Time_Stamp','Relative_Time','HH','LL','H','L','speed','tank1','tank2','Alarm']

    tm = time.ctime() +'_'+cm+ '/'
    path = 'correct_fnn_results/'
    os.mkdir(path+tm)
    os.mkdir(path+tm+'model/')
    train = path+tm+"train.csv"
    test = path+tm+"test.csv"
    #label_train_pre=path+tm+"label_train_pre.csv"
    #label_train_act = path+tm+"label_train_act.csv"
    #label_test_pre = path + tm+"label_test_pre.csv"
    #label_test_act = path+tm+"label_test_act.csv"
    indices_test = path+tm+"indices_test.txt"
    report = path + tm + "classification_report.txt"
    #indices_train = path +tm+ "indices_train.csv"
    model_path = path+tm+'model/'
    data.columns = headers
    data = data.dropna()
    X_Label = [i for i in data.columns.tolist() if i not in 'Alarm']
    X_train, y_train,X_test,y_test, train_indx,test_indx = pre_split(data, X_Label,0.3)
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=193)

    num_seq = X_train.shape[0]
    #num_sample =[22,50,100,1000,3000,6000,10000,30000,70000,100000,150000,200000,num_seq]
    num_sample = [num_seq]
    for sample_size in num_sample:
        es = EarlyStopping(monitor='loss', min_delta=1e-6,patience = 35)
        print(sample_size,file=open(train,"a"))
        print("Precision",',',"Recall",',',"F1",',',"Accuray",',',"loss",file=open(train,"a"))
        print(sample_size,file=open(test,"a"))
        print("Precision",',',"Recall",',',"F1",',',"Accuray",',',"loss",file=open(test,"a"))
        #print(sample_size,file=open(label_train,"a"))
        #print(sample_size,file=open(label_test,"a"))

        for kfold in range(10):
            print(sample_size)
             
            model = Sequential()
            indx =np.array(random.sample(range(X_train.shape[0]),sample_size))
            print(X_train[indx].shape)
            model.reset_states()
            model.add(Dense(100,activation ='relu'))
            #model.add(Dense(100,activation='relu'))
            #model.add(Dense(20,activation='relu'))
            #model.add(Dense(100,kernel_initializer=initializers.RandomNormal(stddev=0.1),activation ='relu'))

            model.add(Dense(1,activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
            model.fit(X_train[indx], y_train[indx], batch_size = 1000,shuffle=True,epochs=10000,callbacks=[es])
            model.save(model_path+str(sample_size)+'_'+str(kfold)+".h5")
            y_pre_train = model.predict(X_train[indx])
            y_pre_train[y_pre_train>=0.5]=1
            y_pre_train[y_pre_train<0.5]=0

            y_pre_test = model.predict(X_test)
            y_pre_test[y_pre_test>=0.5]=1
            y_pre_test[y_pre_test<0.5]=0

            #print(X_test)
            y_true_train = y_train[indx] 
            y_true_test = y_test
            ptrain,rtrain,ftrain = evaluate(y_pre_train,y_true_train)
            ptest,rtest,ftest = evaluate(y_pre_test,y_true_test)

            atrain = model.evaluate(X_train[indx], y_train[indx],steps=1)
            atest = model.evaluate(X_test,y_test,steps=1)
            acc_train = atrain[1]
            acc_test = atest[1]
            loss_train = atrain[0]
            loss_test = atest[0]
            print(ptest,',',rtest,',',ftest,',',acc_test,',',loss_test,file=open(test,"a"))
            print(ptrain,',',rtrain,',',ftrain,',',acc_train,',',loss_train,file=open(train,"a"))

            '''print("Train_label",file=open(label_train_act,"a"))
            f = open(label_train_act,'a')
            np.savetxt(f,y_true_train,delimiter=',')
            f.close()
            print("Train_predictions_Label",file=open(label_train_pre,"a"))
            f = open(label_train_pre,'a')
            np.savetxt(f,y_pre_train,delimiter=',')
            f.close()'''
            #true label first, predictions second
            if sample_size == num_seq:

                labels = np.zeros((3,y_test.shape[0]))
                labels[0,:]= test_indx+2
                labels[1,:]= y_true_test
                labels[2,:]= np.transpose(y_pre_test)
                f = open(indices_test,'a')
                np.savetxt(f,labels,delimiter=',')
                f.close()


#logistic('Label_Jan27b.csv')
#NN('Label_Jan27b.csv')
#logistic('Mlabel_Mtim.csv')
#logistic_nondos('pcap file/label_AN_3.csv')
os.environ["CUDA_VISIBLE_DEVICES"]="0"
logistic_nondos('pcap file/label_AN_3.csv')

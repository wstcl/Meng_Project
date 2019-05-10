import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense,LSTM
from sklearn import preprocessing
from keras.callbacks import EarlyStopping
import random
import time
import os
import h5py
from keras.utils import to_categorical
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def evaluate(y_pred, y_test):
    pre = precision_score(y_true=y_test,y_pred=y_pred,average='micro')
    rec = recall_score(y_true=y_test,y_pred=y_pred,average='micro')
    f1 = f1_score(y_true=y_test,y_pred=y_pred,average='micro')
    return pre, rec, f1

def data_pre(file,lstm,fnn):
    timestep = 10
    lstm_model = load_model(lstm)
    data=np.loadtxt(file,delimiter=',')
    drop = data.shape[0] % timestep
    for i in range(drop):
        data = np.delete(data, data.shape[0] - i - 1, axis=0)
    true_label = data[:,-1]
    X = data[:,:-1]
    X = preprocessing.scale(X)
    Y= to_categorical(true_label)
    x_l = X.reshape(X.shape[0]//timestep,timestep,X.shape[1])
    #y_l = Y.reshape(Y.shape[0]//timestep,timestep,Y.shape[1])
    l_pre = lstm_model.predict(x_l)
    l_pre = l_pre.reshape((l_pre.shape[0]*timestep,l_pre.shape[2]))
    l_pre = np.argmax(l_pre,axis=1)
    l_pre = to_categorical(l_pre)
    #FNN
    fnn_model = load_model(fnn)
    x_f = X
    f_pre = fnn_model.predict(x_f)
    f_pre = np.argmax(f_pre, axis=1)
    f_pre = to_categorical(f_pre)
    stackX = np.dstack((l_pre, f_pre))
    stackX = np.reshape(stackX, (stackX.shape[0],stackX.shape[1]*stackX.shape[2]))
    print(stackX.shape)
    print(Y.shape)
    return stackX, Y

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

def ensemble():
    path = 'ensemble_model/'
    tm = time.ctime() + '/'
    os.mkdir(path + tm)
    os.mkdir(path + tm + 'model/')
    train = path + tm + "train.csv"
    test = path + tm + "test.csv"
    n_labels = 11
    indices_test = path + tm + "indices_test.csv"

    model_path = path + tm + 'model/'
    X,y = data_pre('pcap file/all_mclass.csv', 'ensemble_model/LSTM.h5', 'ensemble_model/FNN.h5')
    X_train, y_train, X_test, y_test, train_indx, test_indx = pre_split(X, y, test_size=0.3)
    sample_size = X_train.shape[0]
    print(sample_size, file=open(train, "a"))
    print("Precision", ',', "Recall", ',', "F1", ',', "Accuray", ',', "loss", file=open(train, "a"))
    print(sample_size, file=open(test, "a"))
    print("Precision", ',', "Recall", ',', "F1", ',', "Accuray", ',', "loss", file=open(test, "a"))
        
    for kfold in range(10):
        model=Sequential()
        indx = np.array(random.sample(range(X_train.shape[0]), sample_size))
        es = EarlyStopping(monitor='loss', min_delta=1e-6, patience=35)
        model.reset_states()
        #model.add(Dense(100,activation='relu'))
        model.add(Dense(n_labels,activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['categorical_accuracy'])
        model.fit(X_train[indx], y_train[indx], batch_size=1000, shuffle=True, epochs=10000, callbacks=[es])
        y_pre_train = model.predict_classes(X_train[indx])
        y_pre_test = model.predict_classes(X_test)
        # print(X_test)
        y_true_train = np.argmax(y_train[indx], axis=1)
        y_true_test = np.argmax(y_test, axis=1)
        ptrain, rtrain, ftrain = evaluate(y_pre_train, y_true_train)
        ptest, rtest, ftest = evaluate(y_pre_test, y_true_test)

        atrain = model.evaluate(X_train[indx], y_train[indx], steps=1)
        atest = model.evaluate(X_test, y_test, steps=1)
        acc_train = atrain[1]
        acc_test = atest[1]
        loss_train = atrain[0]
        loss_test = atest[0]
        print(ptest, ',', rtest, ',', ftest, ',', acc_test, ',', loss_test, file=open(test, "a"))
        print(ptrain, ',', rtrain, ',', ftrain, ',', acc_train, ',', loss_train, file=open(train, "a"))

ensemble()

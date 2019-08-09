import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import Sequential,load_model
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
from io import StringIO
import pickle
from keras.callbacks import EarlyStopping,ModelCheckpoint
import datetime


def change_dummy(need_define,data):
    for i in need_define:
        if i == "label":
            boolean_names = {'spy.': 3, 'teardrop.': 1, 'portsweep.': 2, 'ftp_write.': 3, 'loadmodule.': 4, 'phf.': 3, 'smurf.': 1, 'satan.': 2,
                             'nmap.': 2, 'rootkit.': 4, 'buffer_overflow.': 4, 'perl.': 4, 'guess_passwd.': 3, 'pod.': 1, 'neptune.': 1, 'normal.': 0,
                             'warezmaster.': 3, 'multihop.': 3, 'warezclient.': 3, 'land.': 1, 'imap.': 3, 'ipsweep.': 2, 'back.': 1}
            data[i] = data[i].map(boolean_names)
            print(boolean_names)
        else:
            id_dummy = data[i].unique().tolist()
            boolean_names = {}
            for j in range(len(id_dummy)):
                boolean_names[id_dummy[j]] = j
            #print(id_dummy)
            data[i] = data[i].map(boolean_names)

def add_NADE(neurons,train_data,validata):
    model = Sequential()
    es = EarlyStopping(monitor='loss', min_delta=1e-6, patience=30)
    model.add(Dense(neurons[0], activation='sigmoid', input_dim=train_data.shape[1],kernel_initializer='he_uniform'))
    for i in neurons[1:]:
        model.add(Dense(i,activation='sigmoid',kernel_initializer='he_uniform'))
    model.add(Dense(train_data.shape[1],activation='sigmoid' ,kernel_initializer='he_uniform'))
    model.compile(loss='mse',optimizer='adam',metrics=['categorical_accuracy'])
    print(model.summary())
    model.fit(train_data, train_data, validation_data=(validata,validata),epochs=10000, batch_size=256,shuffle=True,callbacks=[es])
    model.pop()
    model.save("NDAE1.h5")


def Stack_NADE(NDAE1,neurons, train_data,validata):
    NDAE2 = Sequential()
    es = EarlyStopping(monitor='loss', min_delta=1e-6, patience=30)
    NDAE2_input = NDAE1.predict(train_data)
    NDAE2_valid = NDAE1.predict(validata)
    NDAE2.add(Dense(neurons[0], activation='sigmoid', input_dim=NDAE2_input.shape[1],kernel_initializer='he_uniform'))
    for i in neurons[1:]:
        NDAE2.add(Dense(i,activation='sigmoid',kernel_initializer='he_uniform'))
    NDAE2.add(Dense(NDAE2_input.shape[1],activation='sigmoid',kernel_initializer='he_uniform' ))
    NDAE2.compile(loss='mse', optimizer='adam', metrics=['categorical_accuracy'])
    print(NDAE2.summary())
    NDAE2.fit(NDAE2_input, NDAE2_input, validation_data=(NDAE2_valid,NDAE2_valid),epochs=10000, batch_size=1000, shuffle=True,callbacks=[es])
    NDAE2.pop()
    NDAE2.save('NDAE2.h5')

def preprocess(X,mean,std):
    X = (X-mean)/std
    return X

def x_y_split(csvname):
    data = pd.read_csv(csvname)

    data.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                    'urgent', 'hot',
                    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                    'num_file_creations', 'num_shells',
                    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count',
                    'serror_rate', 'srv_serror_rate',
                    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                    'dst_host_count', 'dst_host_srv_count',
                    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label']
    dummy_list = ['protocol_type', 'service', 'flag', 'label']
    change_dummy(dummy_list, data)
    # data[pd.to_numeric(data['label'],errors='coerce').notnull()]
    data = data.dropna()
    data = data.as_matrix()
    X = data[:, :-1]
    y = data[:, -1]
    return X,y

def data_preprocess(csvname):

    X,y = x_y_split(csvname)
    #sklearn preprocess when input mean and std is 0
    X = preprocessing.minmax_scale(X)

    #manual preprocess when input and std

    print(X.shape)
    NDAE1 = load_model("NDAE1.h5")
    NDAE2 = load_model("NDAE2.h5")

    NDAE2_input = NDAE1.predict(X)
    RF_input = NDAE2.predict(NDAE2_input)
    return RF_input,y
#to generate new NDAEs

def RF_Classifier(X,y):
    depth = [30]
    for d in depth:
        clf = RandomForestClassifier(n_estimators=100, max_depth=d,random_state = 42)
        clf.fit(X, y)
        y_pre = clf.predict(X)
        print("f1 score of max depth="+str(d),f1_score(y_pred=y_pre,y_true=y,average=None))
        print("f1 score macrof max depth="+str(d), f1_score(y_pred=y_pre, y_true=y, average='macro'))
        return clf




X,y = x_y_split('../kddcup10.csv')
X_val,y_val = x_y_split('../corrected.csv')

X = preprocessing.scale(X)
X_val = preprocessing.scale(X_val)
starttime = datetime.datetime.now()

add_NADE([14,28,28],X,X_val)
NDAE1 = load_model('NDAE1.h5')
Stack_NADE(NDAE1,[14,28,28],X,X_val)

endtime = datetime.datetime.now()
print (endtime - starttime)


X,y = data_preprocess('../kddcup10.csv')
RF = RF_Classifier(X,y)
x_test,y_test = data_preprocess('..//corrected.csv')

y_pre = RF.predict(x_test)
print("f1 score of test",f1_score(y_pred=y_pre,y_true=y_test,average=None))
print("f1 score of test weighted", f1_score(y_pred=y_pre, y_true=y_test, average='weighted'))

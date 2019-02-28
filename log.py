import pandas as pd
import numpy as np
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
    P_score = []
    R_score = []
    F_score = []
    for kfold in range(1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = Sequential()
        model.add(Dense(1, input_dim= 12, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(X_train, y_train, batch_size=1000, epochs=130,validation_split=0.2)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])   
        plt.title('Logistic accuracy')     
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train','test'],loc='upper left')
        plt.show()
        y_pre = model.predict_classes(X_test)
        precision, recall, f1 = evaluate(y_pre, y_test)
        P_score.append(precision)
        R_score.append(recall)
        F_score.append(f1)
        print(confusion_matrix(y_test,y_pre))
    P_score = np.array(P_score)
    R_score = np.array(R_score)
    F_score = np.array(F_score)
    print("Precision: ", np.mean(P_score), "+-", np.std(P_score))
    print("Recall: ", np.mean(R_score), "+-", np.std(R_score))
    print("F1_socre: ", np.mean(F_score), "+-", np.std(F_score))


def NN(csvname):
    data = pd.read_csv(csvname)
    headers = ['SourceIp','DestIp','SourcePort','destPort','Seq_num','Trans_Id','funcCode','Refno','Register_data','Exeption_Code','Time_Stamp','Relative_Time','Alarm']
    data.columns = headers
    X_Label = [i for i in data.columns.tolist() if i not in 'Alarm']
    X, y = pre_split(data, X_Label)
    P_score = []
    R_score = []
    F_score = []
    for kfold in range(1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = Sequential()
        model.add(Dense(256, input_dim=12,activation='tanh'))
        model.add(Dense(128,activation='tanh'))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
        history = model.fit(X_train, y_train, batch_size=1000, epochs=250,validation_split=0.2)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])   
        plt.title('MLP accuracy')     
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train','test'],loc='upper left')
        plt.show()
        y_pre = model.predict_classes(X_test)
        precision, recall, f1 = evaluate(y_pre, y_test)
        P_score.append(precision)
        R_score.append(recall)
        F_score.append(f1)
        print(confusion_matrix(y_test,y_pre))

    P_score = np.array(P_score)
    R_score = np.array(R_score)
    F_score = np.array(F_score)
    print("Precision: ", np.mean(P_score), "+-", np.std(P_score))
    print("Recall: ", np.mean(R_score), "+-", np.std(R_score))
    print("F1_socre: ", np.mean(F_score), "+-", np.std(F_score))
NN('Label_Jan27b.csv')

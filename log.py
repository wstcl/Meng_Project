import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import precision_score,recall_score,f1_score

def evaluate(y_pred, y_test):
    pre = precision_score(y_true=y_test,y_pred=y_pred)
    rec = recall_score(y_true=y_test,y_pred=y_pred)
    f1 = f1_score(y_true=y_test,y_pred=y_pred)
    return pre, rec, f1

def pre_split(input_data, Label):
    X = input_data[Label]
    X = np.array(X,dtype=float)
    y = input_data['Alarm']
    y = np.array(y,dtype=int)
    return X, y

def logistic(csvname):
    data = pd.read_csv(csvname)
    X_Label = [i for i in data.columns.tolist() if i not in 'Alarm']
    X, y = pre_split(data, X_Label)
    P_score = []
    R_score = []
    F_score = []
    for kfold in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = Sequential()
        model.add(Dense(1, input_dim= 13, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=50, epochs=10)
        y_pre = model.predict_classes(X_test)
        precision, recall, f1 = evaluate(y_pre, y_test)
        P_score.append(precision)
        R_score.append(recall)
        F_score.append(f1)
    P_score = np.array(P_score)
    R_score = np.array(R_score)
    F_score = np.array(F_score)
    print("Precision: ", np.mean(P_score), "+-", np.std(P_score))
    print("Recall: ", np.mean(R_score), "+-", np.std(R_score))
    print("F1_socre: ", np.mean(F_score), "+-", np.std(F_score))


def NN(csvname):
    data = pd.read_csv(csvname)
    X_Label = [i for i in data.columns.tolist() if i not in 'Alarm']
    X, y = pre_split(data, X_Label)
    P_score = []
    R_score = []
    F_score = []
    for kfold in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = Sequential()
        model.add(Dense(10, input_dim=13,activation='relu'))
        model.add(Dense(5,activation='relu'))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
        model.fit(X_train, y_train,batch_size=50,epochs=50)
        y_pre = model.predict_classes(X_test)

        precision, recall, f1 = evaluate(y_pre, y_test)
        P_score.append(precision)
        R_score.append(recall)
        F_score.append(f1)

    P_score = np.array(P_score)
    R_score = np.array(R_score)
    F_score = np.array(F_score)
    print("Precision: ", np.mean(P_score), "+-", np.std(P_score))
    print("Recall: ", np.mean(R_score), "+-", np.std(R_score))
    print("F1_socre: ", np.mean(F_score), "+-", np.std(F_score))
NN('D:\dos_pcap\\nov8mn_output.csv')
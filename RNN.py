import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_utils
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score,recall_score,f1_score
from keras.models import Sequential
from keras.layers import Dense
'''
csvname = 'C:\\Users\\93621\\Desktop\\S_dos.csv'
data = pd.read_csv(csvname)
headers = ['SourceIp','DestIp','SourcePort','destPort','Trans_Id','funcCode','Refno','Register_data','Exeption_Code','Time_Stamp']
data.columns = headers
new = [ "Alarm" ]
for i in new:
    data[i] = np.nan
i = 0
data = data.fillna('0')
packets_num = len(data['Alarm'])
IPsource = data['SourceIp'].unique().tolist()
IPdest = data['DestIp'].unique().tolist()
IP = list(set(IPsource + IPdest))
for j in range(0, len(IP)):
    data = data.replace(IP[j],j)
while(i < packets_num - 3):
    if data['Exeption_Code'][i] == 1:
        data['Alarm'][i] = 1

    if len(data['Register_data'][i+1]) > 10:
        data['Register_data'][i + 1] = 123
        if len(data['Register_data'][i+3]) > 10:
            data['Alarm'][i:i+3]  = 1
    i = i + 1
print("Processing is over, start training")
'''
csvname = 'C:\\Users\\93621\\Desktop\\LogLabeled.csv'
data = pd.read_csv(csvname)
n_classes = 2
#data.to_csv('C:\\Users\\93621\\Desktop\\LogLabeled.csv')
X = [i for i in data.columns.tolist() if i not in 'Alarm']
X = data[X]
X = X.as_matrix().astype(np.float64)
y = data['Alarm']
y = y.as_matrix().astype(np.int32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

'''features_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[50,10,5], n_classes=2, feature_columns=features_cols)
dnn_clf.fit(X_train, y_train, batch_size=50, steps=100)
from sklearn.metrics import accuracy_score
y_pred = dnn_clf.predict(X_test)
print(accuracy_score(y_test, list(y_pred)))'''
def logistic():
    logreg = LogisticRegression()
    logreg.fit(X_train,y_train)
    y_pred = logreg.predict(X_test)
    print(logreg.score(X_test, y_test))

def NN():
    model = Sequential()
    model.add(Dense(10, input_dim=10,activation='relu'))
    model.add(Dense(5,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    model.fit(X_train, y_train,batch_size=100,epochs=40)
    scores = model.evaluate(X_test,y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
logistic()